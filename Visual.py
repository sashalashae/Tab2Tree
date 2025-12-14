import pandas as pd
import matplotlib.pyplot as plt

# Results data
data = {
    "Model": ["XGBoost", "MLP"],
    "Original Acc": [0.868, 0.851],
    "Renamed Acc": [0.751, 0.751],
    "+Dict Map Acc": [0.868, 0.851],
    "+Embedding Acc": [0.806, 0.756]
}
df = pd.DataFrame(data)
df["Acc Drop (Renamed)"] = df["Original Acc"] - df["Renamed Acc"]
df["Acc Drop (+Embed)"] = df["Original Acc"] - df["+Embedding Acc"]

# Plot 1: Accuracy Comparison
df.set_index("Model")[["Original Acc", "Renamed Acc", "+Dict Map Acc", "+Embedding Acc"]].plot.bar()
plt.title("Model Accuracy Under Schema Shifts")
plt.ylabel("Accuracy")
plt.ylim(0.7, 0.9)
plt.xticks(rotation=0)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Plot 2: Accuracy Drop
df.set_index("Model")[["Acc Drop (Renamed)", "Acc Drop (+Embed)"]].plot.bar(color=["salmon", "skyblue"])
plt.title("Accuracy Drop Due to Schema Shift")
plt.ylabel("Accuracy Drop")
plt.xticks(rotation=0)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
