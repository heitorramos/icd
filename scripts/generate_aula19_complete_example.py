"""Gera as figuras do exemplo completo da Aula 19."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "exemplos/16-correlacao/data/insurance.csv"
OUT = ROOT / "slides/assets/aula19-verossimilhanca-regressao"

NAVY = "#17324d"
TEAL = "#24788f"
ORANGE = "#d95f02"
LIGHT_ORANGE = "#f6c89f"

sns.set_theme(style="whitegrid", context="talk")
df = pd.read_csv(DATA)

X = sm.add_constant(df["age"])
model = sm.OLS(df["charges"], X).fit()
df["fitted"] = model.fittedvalues
df["residual"] = model.resid


# Ajuste e intervalo para a resposta média.
x_grid = np.linspace(df["age"].min(), df["age"].max(), 200)
prediction = model.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)

fig, ax = plt.subplots(figsize=(11.5, 5.8))
ax.scatter(
    df["age"],
    df["charges"],
    s=24,
    alpha=0.22,
    color=TEAL,
    edgecolor="none",
    label="observações",
)
ax.fill_between(
    x_grid,
    prediction["mean_ci_lower"].to_numpy(),
    prediction["mean_ci_upper"].to_numpy(),
    color=LIGHT_ORANGE,
    alpha=0.72,
    label="IC 95% para a média",
)
ax.plot(
    x_grid,
    prediction["mean"].to_numpy(),
    color=ORANGE,
    linewidth=3.2,
    label="reta estimada",
)
ax.set(
    xlabel="idade (anos)",
    ylabel="despesas médicas (US$)",
    title="Ajuste linear e incerteza sobre a média condicional",
)
ax.legend(frameon=True, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "exemplo-completo-ajuste.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# Comparações didáticas: padrão ideal versus diagnóstico observado.
rng_ideal = np.random.default_rng(1904)
ideal_residual = rng_ideal.normal(0, df["residual"].std(ddof=1), len(df))
ideal_magnitude = np.abs(ideal_residual)
observed_magnitude = np.abs(model.resid)


def binned_trend(x_values, y_values, q=12):
    """Médias em quantis de x para destacar a tendência sem impor uma curva."""
    frame = pd.DataFrame({"x": np.asarray(x_values), "y": np.asarray(y_values)})
    groups = pd.qcut(frame["x"], q=q, duplicates="drop")
    return frame.groupby(groups, observed=True).mean()


fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.4), sharex=True, sharey=True)
axes[0].scatter(df["fitted"], ideal_residual, s=24, alpha=0.32, color=TEAL, edgecolor="none")
axes[0].axhline(0, color=NAVY, linestyle="--", linewidth=1.7)
axes[0].set(
    title="Padrão ideal: nuvem aleatória ao redor de zero",
    xlabel="valor ajustado (US$)",
    ylabel="resíduo (US$)",
)
sns.scatterplot(
    data=df,
    x="fitted",
    y="residual",
    hue="smoker",
    palette={"no": TEAL, "yes": ORANGE},
    alpha=0.50,
    s=32,
    edgecolor="none",
    ax=axes[1],
)
axes[1].axhline(0, color=NAVY, linestyle="--", linewidth=1.7)
axes[1].set(
    title="Dados observados: grupos em patamares distintos",
    xlabel="valor ajustado (US$)",
    ylabel="",
)
axes[1].legend(title="fumante", frameon=True)
fig.tight_layout()
fig.savefig(OUT / "diagnostico-residuos-ajustados.png", dpi=180, bbox_inches="tight")
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), sharex=True, sharey=True)
for ax, sample, title in [
    (axes[0], ideal_residual, "Padrão ideal: pontos próximos da reta"),
    (axes[1], model.resid, "Dados observados: forte desvio da reta"),
]:
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(sample, dist="norm", fit=True)
    ax.scatter(theoretical, ordered, s=22, alpha=0.48, color=TEAL, edgecolor="none")
    reference_x = np.array([-4.3, 4.3])
    ax.plot(
        reference_x,
        intercept + slope * reference_x,
        color=ORANGE,
        linestyle="--",
        linewidth=2.2,
    )
    ax.set(title=title, xlabel="quantis teóricos da Normal")
axes[0].set_ylabel("resíduos ordenados (US$)")
axes[1].set_ylabel("")
axes[0].set_xlim(-4.3, 4.3)
fig.tight_layout()
fig.savefig(OUT / "diagnostico-qq.png", dpi=180, bbox_inches="tight")
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.4), sharex=True, sharey=True)
axes[0].scatter(df["fitted"], ideal_magnitude, s=24, alpha=0.28, color=TEAL, edgecolor="none")
ideal_trend = binned_trend(df["fitted"], ideal_magnitude)
axes[0].plot(ideal_trend["x"], ideal_trend["y"], color=NAVY, linewidth=3)
axes[0].set(
    title="Padrão ideal: tamanho médio aproximadamente constante",
    xlabel="valor ajustado (US$)",
    ylabel="erro absoluto |eᵢ| (US$)",
)

for group, color, label in [("no", TEAL, "não fumante"), ("yes", ORANGE, "fumante")]:
    mask = df["smoker"].eq(group).to_numpy()
    axes[1].scatter(
        df.loc[mask, "fitted"],
        observed_magnitude[mask],
        s=26,
        alpha=0.34,
        color=color,
        edgecolor="none",
        label=label,
    )
    group_trend = binned_trend(df.loc[mask, "fitted"], observed_magnitude[mask], q=8)
    axes[1].plot(group_trend["x"], group_trend["y"], color=color, linewidth=3)
axes[1].set(
    title="Dados observados: escalas diferentes entre grupos",
    xlabel="valor ajustado (US$)",
    ylabel="",
)
axes[1].legend(frameon=True)
fig.tight_layout()
fig.savefig(OUT / "diagnostico-magnitude-erros.png", dpi=180, bbox_inches="tight")
plt.close(fig)
