import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Daten laden ---
df = pd.read_csv("../df_agentic.csv")

# --- RAG-Aufrufe je Frage zählen (nicht-leere 'name_'-Spalten) ---
name_cols = [c for c in df.columns if str(c).startswith("name_")]
df[name_cols] = df[name_cols].applymap(lambda x: np.nan if (isinstance(x, str) and x.strip() == "") else x)
df["rag_calls"] = df[name_cols].notna().sum(axis=1)

# --- Gesamtzahl gezogener Nachbarn Σ m_i berechnen (m_0 ... m_4) ---
m_cols = [c for c in df.columns if str(c).startswith("m_")]
df[m_cols] = df[m_cols].apply(pd.to_numeric, errors="coerce")
df["total_m_neighbors"] = df[m_cols].fillna(0).sum(axis=1)

# --- Fixe Metriken ---
metrics = [
    ("Answer Precision", "answer_precision"),
    ("Answer Recall", "answer_recall"),
    ("Retrieval Precision", "retrieval_precision"),
    ("Retrieval Recall", "retrieval_recall"),
]

# ---------- Plot 1: Metriken vs. Anzahl RAG-Aufrufe ----------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, (label, col) in enumerate(metrics):
    if col not in df.columns:
        continue
    plot_df = df[["rag_calls", col]].dropna()
    if plot_df.empty:
        continue
    plot_df.boxplot(column=col, by="rag_calls", grid=False, ax=axes[i])
    axes[i].set_title(label)
    axes[i].set_xlabel("Anzahl RAG-Aufrufe (pro Frage)")
    axes[i].set_ylabel(label)
    axes[i].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

plt.suptitle("Verteilung der Metriken nach Anzahl RAG-Aufrufe")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out1 = Path("/mnt/data/metrics_boxplots_vs_rag_calls.png")
plt.savefig(out1, bbox_inches="tight", dpi=150)
print(f"Gespeichert: {out1}")

