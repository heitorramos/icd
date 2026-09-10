"""Gera a figura didática sobre cauda pesada e amostras pequenas."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides" / "assets" / "aula10-tcl-inferencia"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0f6b78"
ORANGE = "#d95f02"
INK = "#17324d"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 160, "axes.titleweight": "bold"})

sigma_log = 1.15
x = np.linspace(0.03, 25, 1200)
population_pdf = (
    np.exp(-(np.log(x) ** 2) / (2 * sigma_log**2))
    / (x * sigma_log * np.sqrt(2 * np.pi))
)

# Esta semente produz uma amostra possível que, por acaso, não captura a cauda.
rng = np.random.default_rng(20260228)
small_sample = rng.lognormal(mean=0, sigma=sigma_log, size=20)
larger_sample = rng.lognormal(mean=0, sigma=sigma_log, size=100)

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))

axes[0].plot(x, population_pdf, color=BLUE, linewidth=3)
axes[0].fill_between(x, population_pdf, color=BLUE, alpha=0.22)
axes[0].axvline(np.exp(sigma_log * 2.326), color=ORANGE, linestyle="--", linewidth=2)
axes[0].text(14.9, 0.16, "percentil 99%", color=ORANGE, fontsize=12)
axes[0].set_xlim(0, 25)
axes[0].set_ylim(bottom=0)
axes[0].set_title("População com cauda pesada")
axes[0].set_xlabel("Valor de X")
axes[0].set_ylabel("Densidade")

sample_bins = np.linspace(0, 3, 7)
sns.histplot(
    small_sample, bins=sample_bins, stat="density", color=ORANGE,
    alpha=0.62, edgecolor="white", ax=axes[1],
)
sns.kdeplot(
    small_sample, color=ORANGE, linewidth=3, cut=0, clip=(0, 3), ax=axes[1]
)
axes[1].axvline(small_sample.mean(), color=INK, linestyle="--", linewidth=2)
axes[1].set_xlim(0, 3)
axes[1].set_ylim(bottom=0)
axes[1].set_title("Uma amostra de n = 20")
axes[1].set_xlabel("Valor observado de X")
axes[1].set_ylabel("Densidade")
axes[1].text(
    0.05, 0.94, "A cauda pode não aparecer", transform=axes[1].transAxes,
    color=INK, fontsize=12, fontweight="bold", va="top",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
)

larger_bins = np.linspace(0, 15, 11)
sns.histplot(
    larger_sample, bins=larger_bins, stat="density", color="#6a4c93",
    alpha=0.58, edgecolor="white", ax=axes[2],
)
sns.kdeplot(
    larger_sample, color="#6a4c93", linewidth=3, cut=0, clip=(0, 15), ax=axes[2]
)
axes[2].axvline(larger_sample.mean(), color=INK, linestyle="--", linewidth=2)
axes[2].set_xlim(0, 15)
axes[2].set_ylim(bottom=0)
axes[2].set_title("Outra amostra: n = 100")
axes[2].set_xlabel("Valor observado de X")
axes[2].set_ylabel("Densidade")
axes[2].text(
    0.05, 0.94, "A cauda fica visível", transform=axes[2].transAxes,
    color=INK, fontsize=12, fontweight="bold", va="top",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
)

# O título já aparece no slide; removê-lo daqui devolve altura útil aos painéis.
fig.tight_layout(pad=0.35, w_pad=0.6)
fig.savefig(
    OUT / "cauda-pesada-amostras-20-100-large.png",
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)
