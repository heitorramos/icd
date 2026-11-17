"""Gera a figura didática da geometria L0, L1 e L2 da Aula 23."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides/assets/aula23-regularizacao/geometria-l0-l1-l2.png"
BLUE, ORANGE, GREEN = "#1f6f8b", "#d95f02", "#2a9d8f"

plt.style.use("seaborn-v0_8-whitegrid")

u = np.linspace(-3.0, 3.6, 520)
U, V = np.meshgrid(u, u)

# Esta configuração faz a solução Lasso tocar o vértice (2, 0).
center = np.array([2.55, 0.55])
quadratic = np.array([[1.20, 0.15], [0.15, 0.85]])
DU, DV = U - center[0], V - center[1]
loss = (
    quadratic[0, 0] * DU**2
    + 2 * quadratic[0, 1] * DU * DV
    + quadratic[1, 1] * DV**2
)


def quadratic_value(points):
    differences = np.asarray(points) - center
    return np.einsum("...i,ij,...j->...", differences, quadratic, differences)


axis_values = np.linspace(-3.0, 3.6, 3000)
l0_candidates = np.vstack(
    [
        np.column_stack([axis_values, np.zeros_like(axis_values)]),
        np.column_stack([np.zeros_like(axis_values), axis_values]),
    ]
)
l1_radius = 2.0
l1_x = np.linspace(-l1_radius, l1_radius, 2401)
l1_candidates = np.vstack(
    [
        np.column_stack([l1_x, l1_radius - np.abs(l1_x)]),
        np.column_stack([l1_x, -l1_radius + np.abs(l1_x)]),
    ]
)
l2_radius = 1.85
angles = np.linspace(0, 2 * np.pi, 3000)
l2_candidates = np.column_stack(
    [l2_radius * np.cos(angles), l2_radius * np.sin(angles)]
)

constraint_candidates = [l0_candidates, l1_candidates, l2_candidates]
selected_points = [
    points[np.argmin(quadratic_value(points))] for points in constraint_candidates
]
constraint_colors = [BLUE, ORANGE, GREEN]
constraint_titles = [
    r"$L_0$: $\|\beta\|_0\leq 1$",
    r"$L_1$: $\|\beta\|_1\leq t$",
    r"$L_2$: $\|\beta\|_2\leq t$",
]

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
constraint_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
for ax, kind, color, selected in zip(
    constraint_axes, ["L0", "L1", "L2"], constraint_colors, selected_points
):
    touching_level = float(quadratic_value(selected))
    contour_levels = np.array([0.20, 0.45, 0.72, 1.0, 1.45, 2.1]) * touching_level
    ax.contour(U, V, loss, levels=contour_levels, colors="#9fb4c2", linewidths=1.2)
    ax.contour(
        U, V, loss, levels=[touching_level], colors="#263746", linewidths=2.6
    )

    if kind == "L0":
        ax.plot([-3, 3.6], [0, 0], color=color, lw=5, alpha=0.72)
        ax.plot([0, 0], [-3, 3.6], color=color, lw=5, alpha=0.72)
        ax.text(
            -2.72,
            -2.55,
            "conjunto não convexo",
            color=color,
            fontsize=10,
            fontweight="bold",
        )
    elif kind == "L1":
        diamond = np.array(
            [[0, l1_radius], [l1_radius, 0], [0, -l1_radius], [-l1_radius, 0]]
        )
        ax.fill(diamond[:, 0], diamond[:, 1], color=color, alpha=0.16)
        ax.plot(
            np.r_[diamond[:, 0], diamond[0, 0]],
            np.r_[diamond[:, 1], diamond[0, 1]],
            color=color,
            lw=3,
        )
    else:
        ax.fill(l2_candidates[:, 0], l2_candidates[:, 1], color=color, alpha=0.16)
        ax.plot(l2_candidates[:, 0], l2_candidates[:, 1], color=color, lw=3)

    ax.scatter(
        *center,
        marker="*",
        s=135,
        color="#66727c",
        zorder=7,
        label="mínimo sem restrição",
    )
    ax.scatter(
        *selected,
        s=95,
        color="#111111",
        edgecolor="white",
        linewidth=1.2,
        zorder=8,
        label="solução regularizada",
    )
    if kind == "L1":
        ax.annotate(
            r"$\widehat\beta_2=0$",
            xy=selected,
            xytext=(2.58, -0.82),
            color=color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
        )
    ax.set(
        aspect="equal",
        xlim=(-3, 3.6),
        ylim=(-3, 3.6),
        xlabel=r"$\beta_1$",
        ylabel=r"$\beta_2$",
        title=constraint_titles[["L0", "L1", "L2"].index(kind)],
    )

guide = axes[1, 1]
guide.axis("off")
guide.scatter(0.12, 0.78, marker="*", s=180, color="#66727c", transform=guide.transAxes)
guide.text(
    0.22, 0.78, "mínimo sem restrição", transform=guide.transAxes, va="center", fontsize=12
)
guide.scatter(
    0.12,
    0.62,
    s=115,
    color="#111111",
    edgecolor="white",
    linewidth=1.2,
    transform=guide.transAxes,
)
guide.text(
    0.22, 0.62, "solução regularizada", transform=guide.transAxes, va="center", fontsize=12
)
guide.plot([0.07, 0.17], [0.46, 0.46], color="#263746", lw=3, transform=guide.transAxes)
guide.text(
    0.22, 0.46, "primeira curva que toca", transform=guide.transAxes, va="center", fontsize=12
)
guide.text(
    0.07,
    0.25,
    "A forma da região permitida\ndetermina onde o contato ocorre.",
    transform=guide.transAxes,
    va="center",
    fontsize=12.5,
    color="#263746",
    fontweight="bold",
)

fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("Soluções:", dict(zip(["L0", "L1", "L2"], selected_points)))
