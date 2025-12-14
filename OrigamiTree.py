import re
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET

# ----------------------------
# Utilities: JSON/XML -> nodes
# ----------------------------

def json_to_nodes(obj, prefix=""):
    """
    Returns list of nodes: (path, value_str, depth)
    path like: demographics.age
    """
    nodes = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            nodes.extend(json_to_nodes(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            nodes.extend(json_to_nodes(v, path))
    else:
        # leaf
        depth = prefix.count(".") + (1 if prefix else 0)
        nodes.append((prefix, str(obj), depth))
    return nodes

def xml_to_nodes(xml_str):
    """
    Parse XML into nodes: (path, value_str, depth).
    path built from tags along the tree.
    """
    root = ET.fromstring(xml_str)
    nodes = []

    def walk(elem, path="", depth=0):
        cur_path = f"{path}.{elem.tag}" if path else elem.tag
        text = (elem.text or "").strip()
        # If leaf-ish (has text and no children), record
        if text and len(list(elem)) == 0:
            nodes.append((cur_path, text, depth))
        for child in list(elem):
            walk(child, cur_path, depth + 1)

    walk(root, "", 0)
    return nodes

def extract_income_label_from_json(jrec):
    # expects "income" at top-level
    return 1 if str(jrec["income"]).strip() == ">50K" else 0

def extract_income_label_from_xml(xml_str):
    root = ET.fromstring(xml_str)
    inc = root.find(".//income")
    if inc is None:
        # fallback: try Income
        inc = root.find(".//Income")
    val = (inc.text or "").strip() if inc is not None else ""
    return 1 if val == ">50K" else 0

# ---------------------------------------
# Linearization (ORIGAMI-like representation)
# ---------------------------------------

def linearize_nodes(nodes):
    """
    Convert nodes to a single sequence string.
    Example: "demographics.age=39 demographics.sex=Male ..."
    """
    parts = []
    for path, val, _depth in nodes:
        # normalize separators for stability
        p = path.replace("-", "_")
        v = re.sub(r"\s+", "_", str(val).strip())
        parts.append(f"{p}={v}")
    return " ".join(parts)

# ------------------------------------------------
# Key renaming shift + Embedding-based key alignment
# ------------------------------------------------

def apply_key_rename_shift_json(jrec, rename_pairs):
    """
    rename_pairs: dict {old_key: new_key} applied anywhere in JSON keys
    """
    import json
    s = json.dumps(jrec)
    for old, new in rename_pairs.items():
        s = s.replace(f'"{old}"', f'"{new}"')
    return json.loads(s)

def apply_key_rename_shift_xml(xml_str, rename_pairs):
    """
    Simple tag renaming in raw xml string (works if tags appear exactly).
    """
    s = xml_str
    for old, new in rename_pairs.items():
        # replace opening and closing tags
        s = s.replace(f"<{old}>", f"<{new}>").replace(f"</{old}>", f"</{new}>")
    return s

def build_key_vocab_from_nodes(list_of_nodes):
    keys = set()
    for nodes in list_of_nodes:
        for path, _val, _depth in nodes:
            keys.add(path.replace("-", "_"))
    return sorted(keys)

def embedding_key_map(shifted_keys, original_keys, model_name="all-MiniLM-L6-v2"):
    """
    One-to-one greedy matching via sentence transformer embeddings.
    Returns dict mapping shifted_key -> best_original_key.
    """
    st = SentenceTransformer(model_name)
    emb_shift = st.encode(shifted_keys)
    emb_orig = st.encode(original_keys)
    sim = cosine_similarity(emb_shift, emb_orig)

    mapping = {}
    used = set()
    for i, sk in enumerate(shifted_keys):
        # mask used originals
        if used:
            sim[i, list(used)] = -1
        j = int(sim[i].argmax())
        mapping[sk] = original_keys[j]
        used.add(j)
    return mapping

def remap_node_paths(nodes, mapping):
    remapped = []
    for path, val, depth in nodes:
        p = path.replace("-", "_")
        remapped.append((mapping.get(p, p), val, depth))
    return remapped

# ----------------------------
# Tokenization (simple, local)
# ----------------------------

class SimpleVocab:
    def __init__(self):
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def add_text(self, text):
        for tok in text.split():
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text, max_len):
        toks = text.split()
        ids = [self.stoi.get(t, 1) for t in toks][:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

# ----------------------------
# Datasets
# ----------------------------

class LinearizedStructuredDataset(Dataset):
    """
    ORIGAMI-like: linearize all nodes into a token sequence.
    """
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(self.vocab.encode(self.texts[idx], self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

class TreeNodesDataset(Dataset):
    """
    TreeTransformer-like: represent each record as a fixed-size set of nodes.
    We encode each node as token ids for "path=value" plus depth embedding.
    """
    def __init__(self, node_lists, labels, vocab, max_nodes=128, token_len_per_node=6):
        self.node_lists = node_lists
        self.labels = labels
        self.vocab = vocab
        self.max_nodes = max_nodes
        self.token_len_per_node = token_len_per_node

    def __len__(self):
        return len(self.node_lists)

    def __getitem__(self, idx):
        nodes = self.node_lists[idx][:self.max_nodes]
        # Build node token sequences like: ["demographics.age=39"]
        node_texts = []
        depths = []
        for path, val, depth in nodes:
            path = path.replace("-", "_")
            val = re.sub(r"\s+", "_", str(val).strip())
            node_texts.append(f"{path}={val}")
            depths.append(min(depth, 32))

        # Pad nodes
        while len(node_texts) < self.max_nodes:
            node_texts.append("<pad>")
            depths.append(0)

        # Encode each node into a short token sequence (usually 1 token in our simple scheme)
        node_ids = []
        for t in node_texts:
            node_ids.append(self.vocab.encode(t, self.token_len_per_node))
        node_ids = torch.tensor(node_ids, dtype=torch.long)  # (max_nodes, token_len_per_node)
        depths = torch.tensor(depths, dtype=torch.long)       # (max_nodes,)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return node_ids, depths, y

# ----------------------------
# Models
# ----------------------------

class ORIGAMIBaseline(nn.Module):
    """
    ORIGAMI-like: Transformer encoder over a linearized key=value token stream.
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_ff=256, max_len=256, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)
        self.max_len = max_len

    def forward(self, x):
        # x: (B, L)
        B, L = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.enc(h)
        # mean pool over non-pad tokens
        mask = (x != 0).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
        return self.cls(pooled)

class TreeTransformerBaseline(nn.Module):
    """
    TreeTransformer-like: Transformer over node embeddings with depth encodings.
    Node embedding = mean of token embeddings for node_text + depth embedding.
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_ff=256, max_nodes=128, token_len_per_node=6, num_classes=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.depth_embed = nn.Embedding(33, d_model)  # depths capped at 32
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)
        self.max_nodes = max_nodes
        self.token_len_per_node = token_len_per_node

    def forward(self, node_ids, depths):
        # node_ids: (B, N, T), depths: (B, N)
        B, N, T = node_ids.shape
        tok = self.token_embed(node_ids)  # (B, N, T, D)
        tok_mask = (node_ids != 0).float().unsqueeze(-1)
        node_repr = (tok * tok_mask).sum(dim=2) / (tok_mask.sum(dim=2).clamp_min(1.0))  # (B, N, D)
        node_repr = node_repr + self.depth_embed(depths.clamp_max(32))
        h = self.enc(node_repr)  # (B, N, D)
        # mean pool over non-pad nodes
        # pad nodes are those with all-zero node_ids; mark as pad if first token is pad
        node_pad = (node_ids[:, :, 0] == 0).float().unsqueeze(-1)
        keep = 1.0 - node_pad
        pooled = (h * keep).sum(dim=1) / (keep.sum(dim=1).clamp_min(1.0))
        return self.cls(pooled)

# ----------------------------
# Training / Eval
# ----------------------------

def train_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        optim.zero_grad()
        if len(batch) == 2:
            x, y = [b.to(device) for b in batch]
            logits = model(x)
        else:
            node_ids, depths, y = [b.to(device) for b in batch]
            logits = model(node_ids, depths)
        loss = ce(logits, y)
        loss.backward()
        optim.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, preds = [], []
    for batch in loader:
        if len(batch) == 2:
            x, y = [b.to(device) for b in batch]
            logits = model(x)
        else:
            node_ids, depths, y = [b.to(device) for b in batch]
            logits = model(node_ids, depths)
        p = logits.argmax(dim=1).cpu().numpy().tolist()
        preds.extend(p)
        ys.extend(y.cpu().numpy().tolist())
    return accuracy_score(ys, preds), f1_score(ys, preds)

# ----------------------------
# Build hierarchical datasets
# ----------------------------

def build_json_sets(json_data, shift=False, align_fix=False):
    """
    shift: apply key rename shift
    align_fix: apply embedding-based alignment back to original keys (only meaningful if shift=True)
    Returns:
      - orig_nodes_list (for key vocab reference)
      - node_lists (post shift/fix)
      - texts (linearized)
      - labels
    """
    # Define a simple consistent shift for demo (matches your earlier shift)
    rename_pairs = {"workclass": "employment", "marital-status": "maritalstatus"}

    orig_nodes = [json_to_nodes(j) for j in json_data]
    orig_key_vocab = build_key_vocab_from_nodes(orig_nodes)

    if not shift:
        node_lists = orig_nodes
    else:
        shifted = [json_to_nodes(apply_key_rename_shift_json(j, rename_pairs)) for j in json_data]
        if align_fix:
            shifted_keys = build_key_vocab_from_nodes(shifted)
            mapping = embedding_key_map(shifted_keys, orig_key_vocab)
            node_lists = [remap_node_paths(nodes, mapping) for nodes in shifted]
        else:
            node_lists = shifted

    texts = [linearize_nodes(nodes) for nodes in node_lists]
    labels = [extract_income_label_from_json(j) for j in json_data]
    return orig_nodes, node_lists, texts, labels

def build_xml_sets(xml_data, shift=False, align_fix=False):
    rename_pairs = {"workclass": "employment", "marital_status": "maritalstatus"}  # note your XML uses underscores
    orig_nodes = [xml_to_nodes(x) for x in xml_data]
    orig_key_vocab = build_key_vocab_from_nodes(orig_nodes)

    if not shift:
        node_lists = orig_nodes
    else:
        shifted_xml = [apply_key_rename_shift_xml(x, rename_pairs) for x in xml_data]
        shifted = [xml_to_nodes(x) for x in shifted_xml]
        if align_fix:
            shifted_keys = build_key_vocab_from_nodes(shifted)
            mapping = embedding_key_map(shifted_keys, orig_key_vocab)
            node_lists = [remap_node_paths(nodes, mapping) for nodes in shifted]
        else:
            node_lists = shifted

    texts = [linearize_nodes(nodes) for nodes in node_lists]
    labels = [extract_income_label_from_xml(x) for x in xml_data]
    return orig_nodes, node_lists, texts, labels

# ----------------------------
# Run Section 6.4 experiments
# ----------------------------

def run_structured_experiment(json_data, xml_data, epochs=6, batch_size=16, max_len=256, max_nodes=128, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Build three conditions for JSON ----
    _, json_nodes_orig, json_text_orig, json_y = build_json_sets(json_data, shift=False)
    _, json_nodes_shift, json_text_shift, _ = build_json_sets(json_data, shift=True, align_fix=False)
    _, json_nodes_fix, json_text_fix, _ = build_json_sets(json_data, shift=True, align_fix=True)

    # ---- Same for XML ----
    _, xml_nodes_orig, xml_text_orig, xml_y = build_xml_sets(xml_data, shift=False)
    _, xml_nodes_shift, xml_text_shift, _ = build_xml_sets(xml_data, shift=True, align_fix=False)
    _, xml_nodes_fix, xml_text_fix, _ = build_xml_sets(xml_data, shift=True, align_fix=True)

    # Combine JSON + XML to make training a bit more stable (optional)
    # You can also run JSON-only and XML-only by splitting this.
    train_texts = json_text_orig + xml_text_orig
    train_labels = json_y + xml_y

    # Build vocab from training only (original schema)
    vocab = SimpleVocab()
    for t in train_texts:
        vocab.add_text(t)

    # Split into train/test over record indices (keep same split across conditions)
    n = len(train_texts)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    def subset(lst, ix): return [lst[i] for i in ix]

    # ORIGAMI-like datasets (linearized)
    ds_tr = LinearizedStructuredDataset(subset(train_texts, tr_idx), subset(train_labels, tr_idx), vocab, max_len=max_len)
    ds_te_orig = LinearizedStructuredDataset(subset(train_texts, te_idx), subset(train_labels, te_idx), vocab, max_len=max_len)

    # For shifted/fix, we need corresponding test texts in the same order:
    # Build condition-aligned list of texts in combined order
    cond_text_shift = json_text_shift + xml_text_shift
    cond_text_fix = json_text_fix + xml_text_fix

    ds_te_shift = LinearizedStructuredDataset(subset(cond_text_shift, te_idx), subset(train_labels, te_idx), vocab, max_len=max_len)
    ds_te_fix = LinearizedStructuredDataset(subset(cond_text_fix, te_idx), subset(train_labels, te_idx), vocab, max_len=max_len)

    # TreeTransformer-like datasets (nodes)
    train_nodes = json_nodes_orig + xml_nodes_orig
    cond_nodes_shift = json_nodes_shift + xml_nodes_shift
    cond_nodes_fix = json_nodes_fix + xml_nodes_fix

    tree_tr = TreeNodesDataset(subset(train_nodes, tr_idx), subset(train_labels, tr_idx), vocab, max_nodes=max_nodes)
    tree_te_orig = TreeNodesDataset(subset(train_nodes, te_idx), subset(train_labels, te_idx), vocab, max_nodes=max_nodes)
    tree_te_shift = TreeNodesDataset(subset(cond_nodes_shift, te_idx), subset(train_labels, te_idx), vocab, max_nodes=max_nodes)
    tree_te_fix = TreeNodesDataset(subset(cond_nodes_fix, te_idx), subset(train_labels, te_idx), vocab, max_nodes=max_nodes)

    # DataLoaders
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_te_orig = DataLoader(ds_te_orig, batch_size=batch_size)
    dl_te_shift = DataLoader(ds_te_shift, batch_size=batch_size)
    dl_te_fix = DataLoader(ds_te_fix, batch_size=batch_size)

    tree_dl_tr = DataLoader(tree_tr, batch_size=batch_size, shuffle=True)
    tree_dl_te_orig = DataLoader(tree_te_orig, batch_size=batch_size)
    tree_dl_te_shift = DataLoader(tree_te_shift, batch_size=batch_size)
    tree_dl_te_fix = DataLoader(tree_te_fix, batch_size=batch_size)

    # ---- Train ORIGAMI-like ----
    origami = ORIGAMIBaseline(vocab_size=len(vocab.itos), max_len=max_len).to(device)
    opt = torch.optim.AdamW(origami.parameters(), lr=2e-4)

    for ep in range(epochs):
        train_epoch(origami, dl_tr, opt, device)

    acc_o, f1_o = eval_model(origami, dl_te_orig, device)
    acc_s, f1_s = eval_model(origami, dl_te_shift, device)
    acc_f, f1_f = eval_model(origami, dl_te_fix, device)

    print("\n=== ORIGAMI-like (JSON/XML) ===")
    print(f"Original schema:        Acc={acc_o:.3f}  F1={f1_o:.3f}")
    print(f"Shifted schema (no fix):Acc={acc_s:.3f}  F1={f1_s:.3f}")
    print(f"Shifted + key fix:      Acc={acc_f:.3f}  F1={f1_f:.3f}")

    # ---- Train TreeTransformer-like ----
    tree_model = TreeTransformerBaseline(vocab_size=len(vocab.itos), max_nodes=max_nodes).to(device)
    opt2 = torch.optim.AdamW(tree_model.parameters(), lr=2e-4)

    for ep in range(epochs):
        train_epoch(tree_model, tree_dl_tr, opt2, device)

    acc_o2, f1_o2 = eval_model(tree_model, tree_dl_te_orig, device)
    acc_s2, f1_s2 = eval_model(tree_model, tree_dl_te_shift, device)
    acc_f2, f1_f2 = eval_model(tree_model, tree_dl_te_fix, device)

    print("\n=== TreeTransformer-like (JSON/XML) ===")
    print(f"Original schema:        Acc={acc_o2:.3f}  F1={f1_o2:.3f}")
    print(f"Shifted schema (no fix):Acc={acc_s2:.3f}  F1={f1_s2:.3f}")
    print(f"Shifted + key fix:      Acc={acc_f2:.3f}  F1={f1_f2:.3f}")

# ---- Run it ----
# Use your existing json_data and xml_data built earlier.
# NOTE: your current code samples n=100; that is fine for a demo baseline.
run_structured_experiment(json_data, xml_data, epochs=6, batch_size=16, max_len=256, max_nodes=96, seed=0)
