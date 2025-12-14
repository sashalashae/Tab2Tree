# ============================================================
# Hierarchical Schema Robustness: ORIGAMI + TreeTransformer
# ============================================================

import pandas as pd
import numpy as np
import json
import re
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# -------------------------
# 1. Load Adult Dataset
# -------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

df = pd.read_csv(url, names=columns, sep=r",\s*", engine="python")
df = df.replace("?", np.nan).dropna()
df = df.drop(columns=["fnlwgt","education-num"])

# -------------------------
# 2. JSON Conversion
# -------------------------

json_data = []
labels = []

for _, r in df.sample(n=100, random_state=42).iterrows():
    rec = r.to_dict()
    json_data.append({
        "demographics":{
            "age":rec["age"],
            "sex":rec["sex"],
            "race":rec["race"],
            "marital-status":rec["marital-status"]
        },
        "education":rec["education"],
        "work":{
            "workclass":rec["workclass"],
            "hours-per-week":rec["hours-per-week"]
        },
        "income":rec["income"]
    })
    labels.append(1 if rec["income"] == ">50K" else 0)

# -------------------------
# 3. JSON → Tree Nodes
# -------------------------

def json_to_nodes(obj, prefix="", depth=0):
    nodes = []
    if isinstance(obj, dict):
        for k,v in obj.items():
            nodes.extend(json_to_nodes(v, f"{prefix}.{k}" if prefix else k, depth+1))
    else:
        nodes.append((prefix.replace("-","_"), str(obj), depth))
    return nodes

nodes = [json_to_nodes(j) for j in json_data]

# -------------------------
# 4. Linearization (ORIGAMI)
# -------------------------

def linearize(nodes):
    return " ".join([f"{k}={v}" for k,v,_ in nodes])

texts = [linearize(n) for n in nodes]

# -------------------------
# 5. Vocabulary
# -------------------------

class Vocab:
    def __init__(self):
        self.stoi = {"<pad>":0}
        self.itos = ["<pad>"]

    def add(self, text):
        for t in text.split():
            if t not in self.stoi:
                self.stoi[t] = len(self.itos)
                self.itos.append(t)

    def encode(self, text, L):
        ids = [self.stoi.get(t,0) for t in text.split()][:L]
        return ids + [0]*(L-len(ids))

vocab = Vocab()
for t in texts:
    vocab.add(t)

# -------------------------
# 6. Train/Test Split
# -------------------------

Xtr, Xte, ytr, yte = train_test_split(
    list(range(len(nodes))), labels, test_size=0.2, random_state=0
)

# -------------------------
# 7. ORIGAMI Dataset
# -------------------------

class ORIGAMIDS(Dataset):
    def __init__(self, idxs, nodes, labels, vocab, L=256):
        self.X = [vocab.encode(linearize(nodes[i]), L) for i in idxs]
        self.y = [labels[i] for i in idxs]

    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])

# -------------------------
# 8. Tree Dataset
# -------------------------

class TreeDS(Dataset):
    def __init__(self, idxs, nodes, labels, vocab, max_nodes=64):
        self.nodes = nodes
        self.labels = labels
        self.idxs = idxs
        self.vocab = vocab
        self.max_nodes = max_nodes

    def __len__(self): return len(self.idxs)

    def __getitem__(self,i):
        idx = self.idxs[i]
        node_ids, depths = [], []
        for k,v,d in self.nodes[idx][:self.max_nodes]:
            tok = f"{k}={v}"
            node_ids.append(self.vocab.encode(tok,1))
            depths.append(min(d,31))
        while len(node_ids) < self.max_nodes:
            node_ids.append([0])
            depths.append(0)
        return (
            torch.tensor(node_ids),
            torch.tensor(depths),
            torch.tensor(self.labels[idx])
        )

# -------------------------
# 9. ORIGAMI Model
# -------------------------

class ORIGAMI(nn.Module):
    def __init__(self, vocab_size, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        enc = nn.TransformerEncoderLayer(d, 4, 256, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 2)
        self.fc = nn.Linear(d,2)

    def forward(self,x):
        h = self.tr(self.emb(x))
        return self.fc(h.mean(1))

# -------------------------
# 10. TreeTransformer Model
# -------------------------

class TreeTransformer(nn.Module):
    def __init__(self, vocab_size, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.depth = nn.Embedding(32, d)
        enc = nn.TransformerEncoderLayer(d, 4, 256, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 2)
        self.fc = nn.Linear(d,2)

    def forward(self,x,depth):
        h = self.emb(x).mean(2) + self.depth(depth)
        return self.fc(self.tr(h).mean(1))

# -------------------------
# 11. Training Helpers
# -------------------------

def train_epoch(model, loader, opt, tree=False):
    model.train()
    for batch in loader:
        opt.zero_grad()
        if tree:
            x,d,y = batch
            out = model(x,d)
        else:
            x,y = batch
            out = model(x)
        loss = nn.CrossEntropyLoss()(out,y)
        loss.backward()
        opt.step()

def evaluate(model, loader, tree=False):
    model.eval()
    y,p = [],[]
    with torch.no_grad():
        for batch in loader:
            if tree:
                x,d,t = batch
                out = model(x,d)
            else:
                x,t = batch
                out = model(x)
            p.extend(out.argmax(1).numpy())
            y.extend(t.numpy())
    return accuracy_score(y,p), f1_score(y,p)

# -------------------------
# 12. Run ORIGAMI
# -------------------------

origami_tr = DataLoader(ORIGAMIDS(Xtr,nodes,labels,vocab), batch_size=16, shuffle=True)
origami_te = DataLoader(ORIGAMIDS(Xte,nodes,labels,vocab), batch_size=16)

origami = ORIGAMI(len(vocab.itos))
opt_o = torch.optim.AdamW(origami.parameters(), 2e-4)

for _ in range(6):
    train_epoch(origami, origami_tr, opt_o)

acc_o, f1_o = evaluate(origami, origami_te)
print("ORIGAMI JSON Accuracy:", acc_o, "F1:", f1_o)

# -------------------------
# 13. Run TreeTransformer
# -------------------------

tree_tr = DataLoader(TreeDS(Xtr,nodes,labels,vocab), batch_size=16, shuffle=True)
tree_te = DataLoader(TreeDS(Xte,nodes,labels,vocab), batch_size=16)

tree = TreeTransformer(len(vocab.itos))
opt_t = torch.optim.AdamW(tree.parameters(), 2e-4)

for _ in range(6):
    train_epoch(tree, tree_tr, opt_t, tree=True)

acc_t, f1_t = evaluate(tree, tree_te, tree=True)
print("TreeTransformer JSON Accuracy:", acc_t, "F1:", f1_t)
