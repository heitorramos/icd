"""Gera a comparação bootstrap do exemplo A/B da Olist usado nas Aulas 11 e 12."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides/assets/aula12-bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0f6b78"
ORANGE = "#d95f02"
INK = "#17324d"
LIGHT = "#cbd5da"

rng = np.random.default_rng(2026)
R = 50_000
n_a = n_b = 5_000
x_a, x_b = 415, 345
p_a, p_b = x_a / n_a, x_b / n_b
delta_obs = p_b - p_a

# Uma linha por pedido: 1 indica atraso e 0 indica entrega no prazo.
resultados_a = np.r_[np.ones(x_a, dtype=np.int8), np.zeros(n_a - x_a, dtype=np.int8)]
resultados_b = np.r_[np.ones(x_b, dtype=np.int8), np.zeros(n_b - x_b, dtype=np.int8)]

# Bootstrap usual: reamostramos, com reposição, os pedidos de cada grupo.
boot_effect = np.empty(R)
for r in range(R):
    amostra_a = rng.choice(resultados_a, size=n_a, replace=True)
    amostra_b = rng.choice(resultados_b, size=n_b, replace=True)
    boot_effect[r] = amostra_b.mean() - amostra_a.mean()
ci = np.quantile(boot_effect, [0.025, 0.975])

# Bootstrap sob H0: reamostramos dois grupos da amostra empírica combinada.
p_pool = (x_a + x_b) / (n_a + n_b)
resultados_pool = np.r_[resultados_a, resultados_b]
boot_null = np.empty(R)
for r in range(R):
    amostra_a_0 = rng.choice(resultados_pool, size=n_a, replace=True)
    amostra_b_0 = rng.choice(resultados_pool, size=n_b, replace=True)
    boot_null[r] = amostra_b_0.mean() - amostra_a_0.mean()
extreme = boot_null <= delta_obs
p_value = (1 + extreme.sum()) / (R + 1)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), sharey=True)

axes[0].hist(100 * boot_effect, bins=55, color=BLUE, edgecolor="white")
axes[0].axvline(100 * delta_obs, color=ORANGE, linewidth=3)
axes[0].axvspan(100 * ci[0], 100 * ci[1], color=BLUE, alpha=0.14)
axes[0].axvline(0, color=INK, linestyle="--", linewidth=1.5)
axes[0].set_title("Bootstrap usual: incerteza do efeito", fontweight="bold")
axes[0].set_xlabel("Diferença B − A (p.p.)")
axes[0].set_ylabel("Réplicas")
axes[0].text(
    0.04,
    0.94,
    f"IC 95%: [{100*ci[0]:.2f}; {100*ci[1]:.2f}] p.p.",
    transform=axes[0].transAxes,
    ha="left",
    va="top",
    color=INK,
    fontweight="bold",
)

axes[1].hist(100 * boot_null, bins=55, color=LIGHT, edgecolor="white")
axes[1].hist(
    100 * boot_null[extreme],
    bins=20,
    color=ORANGE,
    edgecolor="white",
)
axes[1].axvline(100 * delta_obs, color=ORANGE, linewidth=3)
axes[1].axvline(0, color=INK, linestyle="--", linewidth=1.5)
axes[1].set_title("Bootstrap sob H$_0$: valor-p", fontweight="bold")
axes[1].set_xlabel("Diferença B − A (p.p.)")
axes[1].text(
    0.96,
    0.94,
    f"p ≈ {p_value:.4f}",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    color=INK,
    fontweight="bold",
)

fig.suptitle(
    "O bootstrap usual estima o efeito; o bootstrap sob H$_0$ mede sua extremidade",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT / "ab-olist-bootstrap.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"efeito={delta_obs:.6f}")
print(f"IC95%=[{ci[0]:.6f}, {ci[1]:.6f}]")
print(f"extremas={extreme.sum()}; p={p_value:.6f}")
