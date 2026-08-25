"""Gera as figuras das Aulas 09 e 09 a partir de Palmer Penguins."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tmp" / "pdfs" / "penguins.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    "main/inst/extdata/penguins.csv"
)
OUT = ROOT / "slides" / "assets" / "aula08-estimadores"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"Adelie": "#0f6b78", "Chinstrap": "#d95f02", "Gentoo": "#6a4c93"}
BLUE = "#0f6b78"
ORANGE = "#d95f02"
INK = "#17324d"
MUTED = "#73808a"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 160, "axes.titleweight": "bold"})

data_source = DATA if DATA.exists() else DATA_URL
penguins = pd.read_csv(data_source).dropna(subset=["body_mass_g", "species", "island"])
masses = penguins["body_mass_g"].to_numpy()
mu = masses.mean()
rng = np.random.default_rng(20260824)


def finish(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close()


# 1. População completa e parâmetro.
plt.figure(figsize=(9, 5.2))
sns.histplot(data=penguins, x="body_mass_g", hue="species", bins=22,
             palette=COLORS, multiple="stack", edgecolor="white")
plt.axvline(mu, color=INK, lw=3, label=f"média da população = {mu:,.0f} g")
plt.xlabel("Massa corporal (g)")
plt.ylabel("Número de pinguins")
plt.title("A população que queremos resumir")
plt.legend(title="")
finish("populacao-massa.png")

# 2. Uma amostra aleatória sobre a população.
sample_idx = rng.choice(penguins.index.to_numpy(), size=20, replace=False)
ordered = penguins.sort_values("body_mass_g").reset_index()
ordered["sample"] = ordered["index"].isin(sample_idx)
plt.figure(figsize=(9, 4.8))
plt.scatter(ordered["body_mass_g"], np.zeros(len(ordered)), s=18, color="#cbd5da", alpha=.75)
chosen = ordered[ordered["sample"]]
plt.scatter(chosen["body_mass_g"], np.zeros(len(chosen)), s=70, color=ORANGE,
            edgecolor="white", linewidth=.8, label="amostra (n = 20)")
plt.axvline(mu, color=INK, lw=2.5, label=f"μ = {mu:,.0f} g")
plt.axvline(chosen["body_mass_g"].mean(), color=ORANGE, lw=2.5, ls="--",
            label=f"x̄ = {chosen['body_mass_g'].mean():,.0f} g")
plt.yticks([])
plt.xlabel("Massa corporal (g)")
plt.title("Uma amostra revela apenas parte da população")
plt.legend(ncol=3, loc="upper center")
finish("uma-amostra.png")

# Simulações reutilizadas.
def sample_means(values, n, reps=4000):
    return np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(reps)])

means_10 = sample_means(masses, 10)
means_30 = sample_means(masses, 30)
means_80 = sample_means(masses, 80)

# 3. Muitas amostras e distribuição amostral.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for i in range(10):
    sample = rng.choice(masses, size=20, replace=True)
    axes[0].scatter(sample, np.full(20, i), s=13, color=BLUE, alpha=.55)
    axes[0].scatter(sample.mean(), i, s=65, color=ORANGE, edgecolor="white")
axes[0].axvline(mu, color=INK, lw=2)
axes[0].set_yticks([])
axes[0].set_xlabel("Massa (g)")
axes[0].set_title("Cada amostra produz uma média")
sns.histplot(means_30, bins=28, color=BLUE, ax=axes[1])
axes[1].axvline(mu, color=ORANGE, lw=3)
axes[1].set_xlabel("Média amostral (g)")
axes[1].set_ylabel("Frequência")
axes[1].set_title("As médias formam uma distribuição")
finish("distribuicao-amostral.png")

# 4. Tamanho de amostra.
plt.figure(figsize=(9, 5.2))
for values, label, color in [(means_10, "n = 10", ORANGE), (means_30, "n = 30", BLUE),
                             (means_80, "n = 80", "#6a4c93")]:
    sns.kdeplot(values, label=label, color=color, fill=False, linewidth=3)
plt.axvline(mu, color=INK, lw=2, ls="--", label="μ")
plt.xlabel("Média amostral (g)")
plt.ylabel("Densidade")
plt.title("Aumentar n concentra as estimativas")
plt.legend()
finish("tamanho-amostra.png")

# 5. Erro padrão empírico e teórico.
ns = np.array([5, 10, 20, 30, 50, 80, 120])
empirical_se = np.array([sample_means(masses, int(n), 2500).std(ddof=1) for n in ns])
theoretical_se = masses.std(ddof=0) / np.sqrt(ns)
plt.figure(figsize=(8.5, 5.2))
plt.plot(ns, empirical_se, "o-", lw=3, ms=8, color=ORANGE, label="simulação")
plt.plot(ns, theoretical_se, "--", lw=3, color=BLUE, label="σ / √n")
plt.xlabel("Tamanho da amostra (n)")
plt.ylabel("Erro padrão da média (g)")
plt.title("O erro padrão diminui com a raiz de n")
plt.legend()
finish("erro-padrao.png")

# 6. Amostragem enviesada por espécie.
gentoo = penguins.loc[penguins["species"] == "Gentoo", "body_mass_g"].to_numpy()
biased_means = sample_means(gentoo, 30)
plt.figure(figsize=(9, 5.2))
sns.kdeplot(means_30, fill=True, alpha=.25, color=BLUE, lw=3, label="amostra de toda a população")
sns.kdeplot(biased_means, fill=True, alpha=.20, color=ORANGE, lw=3, label="amostra apenas de Gentoo")
plt.axvline(mu, color=INK, lw=2.5, ls="--", label=f"μ = {mu:,.0f} g")
plt.xlabel("Média amostral (g)")
plt.ylabel("Densidade")
plt.title("Uma amostra maior não corrige seleção enviesada")
plt.legend()
finish("amostragem-enviesada.png")

# 7. TCL: população assimétrica versus médias.
right_skew = np.exp(rng.normal(0, .8, 20000))
right_skew = right_skew / right_skew.mean() * 100
clt_means = sample_means(right_skew, 30, 5000)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
sns.histplot(right_skew, bins=45, color=ORANGE, ax=axes[0])
axes[0].set_xlim(0, 500)
axes[0].set_title("População assimétrica")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Frequência")
sns.histplot(clt_means, bins=35, color=BLUE, ax=axes[1])
axes[1].set_title("Médias de amostras com n = 30")
axes[1].set_xlabel("x̄")
axes[1].set_ylabel("Frequência")
finish("tcl-assimetria.png")

# 8. Média e mediana sob duas perdas.
example = np.array([3100, 3300, 3500, 3700, 3900, 4200, 6100])
candidates = np.linspace(2800, 6400, 500)
mse = np.array([np.mean((example - t) ** 2) for t in candidates])
mae = np.array([np.mean(np.abs(example - t)) for t in candidates])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
axes[0].plot(candidates, mse / 1e6, color=BLUE, lw=3)
axes[0].axvline(example.mean(), color=ORANGE, lw=2.5, ls="--")
axes[0].set_title("Erro quadrático → média")
axes[0].set_ylabel("Perda média (milhões)")
axes[1].plot(candidates, mae, color="#6a4c93", lw=3)
axes[1].axvline(np.median(example), color=ORANGE, lw=2.5, ls="--")
axes[1].set_title("Erro absoluto → mediana")
axes[1].set_ylabel("Perda média (g)")
for ax in axes:
    ax.set_xlabel("Estimativa θ̂ (g)")
finish("funcoes-perda.png")

# 9. Robustez a outlier.
base = penguins["body_mass_g"].dropna().sample(24, random_state=7).to_numpy()
outliers = np.arange(6000, 16001, 500)
mean_values = np.array([np.append(base, x).mean() for x in outliers])
median_values = np.array([np.median(np.append(base, x)) for x in outliers])
plt.figure(figsize=(9, 5.2))
plt.plot(outliers, mean_values, lw=3, color=ORANGE, label="média")
plt.plot(outliers, median_values, lw=3, color=BLUE, label="mediana")
plt.xlabel("Massa do valor extremo adicionado (g)")
plt.ylabel("Estimativa de centro (g)")
plt.title("A média acompanha o outlier; a mediana resiste")
plt.legend()
finish("robustez-outlier.png")

# 10. Trade-off viés-variância com estimadores da média.
reps = 5000
samples = rng.choice(masses, size=(reps, 20), replace=True)
estimators = {
    "média": samples.mean(axis=1),
    "mediana": np.median(samples, axis=1),
    "média aparada 10%": np.sort(samples, axis=1)[:, 2:-2].mean(axis=1),
}
rows = []
for name, vals in estimators.items():
    bias = vals.mean() - mu
    variance = vals.var()
    rows.append((name, bias, variance, bias ** 2 + variance))
metrics = pd.DataFrame(rows, columns=["estimador", "vies", "variancia", "mse"])
x = np.arange(len(metrics))
plt.figure(figsize=(9, 5.2))
plt.bar(x, metrics["variancia"], color=BLUE, label="variância")
plt.bar(x, metrics["vies"] ** 2, bottom=metrics["variancia"], color=ORANGE, label="viés²")
plt.xticks(x, metrics["estimador"])
plt.ylabel("Erro quadrático esperado")
plt.title("O risco combina viés e variância")
plt.legend()
finish("risco-estimadores.png")

# 11. Dispersão dos três estimadores.
plt.figure(figsize=(9, 5.2))
for (name, vals), color in zip(estimators.items(), [BLUE, ORANGE, "#6a4c93"]):
    sns.kdeplot(vals, lw=3, label=name, color=color)
plt.axvline(mu, color=INK, lw=2, ls="--", label="μ")
plt.xlabel("Estimativa da massa média (g)")
plt.ylabel("Densidade")
plt.title("Estimadores diferentes produzem distribuições diferentes")
plt.legend()
finish("comparacao-estimadores.png")

# 12. Distribuição amostral de uma proporção.
p_gentoo = (penguins["species"] == "Gentoo").mean()
prop_samples = np.array([
    (rng.choice(penguins["species"].to_numpy(), size=40, replace=True) == "Gentoo").mean()
    for _ in range(5000)
])
plt.figure(figsize=(9, 5.2))
sns.histplot(prop_samples, bins=np.arange(0.1, 0.66, 0.025), color=BLUE)
plt.axvline(p_gentoo, color=ORANGE, lw=3, label=f"p = {p_gentoo:.3f}")
plt.xlabel("Proporção de Gentoo na amostra")
plt.ylabel("Frequência")
plt.title("Uma proporção também tem distribuição amostral")
plt.legend()
finish("proporcao-gentoo.png")

# 13. Convergência da média acumulada.
sequence = rng.choice(masses, size=2500, replace=True)
running_mean = np.cumsum(sequence) / np.arange(1, len(sequence) + 1)
plt.figure(figsize=(9, 5.2))
plt.plot(np.arange(1, len(sequence) + 1), running_mean, color=BLUE, lw=2)
plt.axhline(mu, color=ORANGE, lw=3, ls="--", label=f"μ = {mu:,.0f} g")
plt.xscale("log")
plt.xlabel("Número de observações (escala log)")
plt.ylabel("Média acumulada (g)")
plt.title("A média se estabiliza à medida que os dados chegam")
plt.legend()
finish("convergencia-media.png")

# 14. Perda assimétrica.
errors = np.linspace(-3, 3, 500)
symmetric = errors ** 2
asymmetric = np.where(errors < 0, 3 * errors ** 2, errors ** 2)
plt.figure(figsize=(9, 5.2))
plt.plot(errors, symmetric, color=BLUE, lw=3, label="quadrática simétrica")
plt.plot(errors, asymmetric, color=ORANGE, lw=3, label="subestimar custa 3×")
plt.axvline(0, color=INK, lw=1.5)
plt.xlabel("Erro = estimativa − valor real")
plt.ylabel("Perda")
plt.title("O contexto pode tornar os erros assimétricos")
plt.legend()
finish("perda-assimetrica.png")

# 15. Quatro combinações de viés e variância.
fig, axes = plt.subplots(2, 2, figsize=(8, 7.2), sharex=True, sharey=True)
settings = [
    (0, .25, "baixo viés\nbaixa variância"),
    (0, .75, "baixo viés\nalta variância"),
    (1.2, .25, "alto viés\nbaixa variância"),
    (1.2, .75, "alto viés\nalta variância"),
]
for ax, (center, spread, title) in zip(axes.flat, settings):
    vals = rng.normal(center, spread, 45)
    ax.scatter(vals, rng.normal(0, .05, len(vals)), s=28, color=BLUE, alpha=.75)
    ax.axvline(0, color=ORANGE, lw=3)
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xlim(-2.2, 2.8)
fig.supxlabel("erro da estimativa (alvo em zero)")
fig.suptitle("Viés desloca; variância espalha", fontweight="bold")
finish("vies-variancia-quadrantes.png")

print(f"Figuras geradas em {OUT}")
print(metrics.round(1).to_string(index=False))
