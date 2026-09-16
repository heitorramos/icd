"""Gera a figura do exemplo A/B didático da Aula 11."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides/assets/aula11-testes-hipoteses"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0f6b78"
ORANGE = "#d95f02"
INK = "#17324d"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 160, "axes.titleweight": "bold"})

grupos = ["A — processo atual", "B — alerta preventivo"]
taxas = 100 * np.array([415 / 5000, 345 / 5000])

fig, ax = plt.subplots(figsize=(9.2, 5.2))
barras = ax.bar(grupos, taxas, color=[ORANGE, BLUE], width=0.58)
ax.set_ylim(0, 10)
ax.set_ylabel("Pedidos atrasados (%)")
ax.set_title("O alerta reduziu a taxa de atraso no exemplo didático")
ax.spines[["top", "right"]].set_visible(False)

for barra, taxa in zip(barras, taxas):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        taxa + 0.25,
        ("{:.1f}%".format(taxa)).replace(".", ","),
        color=INK,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

ax.annotate(
    "redução de 1,4 p.p.",
    xy=(1, taxas[1]),
    xytext=(0.52, 9.25),
    arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.8},
    color=INK,
    ha="center",
    fontweight="bold",
)
ax.text(
    0.5,
    -0.18,
    "Cenário simulado: 5.000 pedidos por grupo",
    transform=ax.transAxes,
    color="#52697d",
    ha="center",
    fontsize=12,
)

fig.tight_layout()
fig.savefig(OUT / "ab-olist-atrasos.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
