"""Gera figuras das Aulas 12 e 13 com a base Marketing A/B Testing."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/marketing-ab/marketing_AB.csv"
DATA_URL = "https://raw.githubusercontent.com/JSRSINGH/Marketing_A-B_Testing_Analysis_Dashboard/main/data/marketing_AB.csv"
OUT = ROOT / "slides/assets/aula12-permutacao"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, PURPLE, INK, LIGHT = "#0f6b78", "#d95f02", "#6a4c93", "#17324d", "#cbd5da"
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 160, "axes.titleweight": "bold"})
rng = np.random.default_rng(1213)


def finish(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close()


df = pd.read_csv(DATA if DATA.exists() else DATA_URL).drop(columns=["Unnamed: 0"])
df["grupo"] = df["test group"].map({"psa": "Controle (PSA)", "ad": "Tratamento (anúncio)"})
summary = df.groupby("grupo", observed=True).agg(n=("converted", "size"), taxa=("converted", "mean"))
observed = summary.loc["Tratamento (anúncio)", "taxa"] - summary.loc["Controle (PSA)", "taxa"]

# 1. Tamanho dos grupos.
counts = df["grupo"].value_counts().reindex(["Controle (PSA)", "Tratamento (anúncio)"])
plt.figure(figsize=(9, 5.2))
counts.plot.bar(color=[ORANGE, BLUE])
plt.xticks(rotation=0)
plt.ylabel("Usuários")
plt.title("A atribuição foi muito desbalanceada")
for i, value in enumerate(counts):
    plt.text(i, value, f"{value:,}".replace(",", "."), ha="center", va="bottom", fontweight="bold")
finish("tamanho-grupos.png")

# 2. Conversão observada.
rates = summary["taxa"].reindex(["Controle (PSA)", "Tratamento (anúncio)"])
plt.figure(figsize=(9, 5.2))
(100 * rates).plot.bar(color=[ORANGE, BLUE])
plt.xticks(rotation=0)
plt.ylabel("Conversão (%)")
plt.title("O anúncio elevou a conversão observada")
for i, value in enumerate(100 * rates):
    plt.text(i, value, f"{value:.2f}%", ha="center", va="bottom", fontweight="bold")
finish("conversao-observada.png")

# 3. Distribuição de exposições.
plot_df = df[df["total ads"] <= df["total ads"].quantile(.99)]
plt.figure(figsize=(9.4, 5.2))
sns.ecdfplot(data=plot_df, x="total ads", hue="grupo", palette=[ORANGE, BLUE], lw=2.5)
plt.xlabel("Total de exposições")
plt.ylabel("Proporção acumulada")
plt.title("As distribuições de exposição são semelhantes")
finish("ecdf-exposicoes.png")

# 4. Conversão por dia.
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
by_day = df.groupby(["most ads day", "grupo"], observed=True)["converted"].mean().mul(100).unstack()
by_day = by_day.reindex(day_order)
by_day.plot(figsize=(10, 5.2), marker="o", lw=2.5, color=[ORANGE, BLUE])
plt.xlabel("")
plt.ylabel("Conversão (%)")
plt.title("A diferença aparece em todos os dias")
plt.legend(title="")
finish("conversao-dia.png")

# Permutações vetorizadas via hipergeométrica: sob H0, conversões são redistribuídas.
n_ad = int((df["test group"] == "ad").sum())
n_total = len(df)
n_conv = int(df["converted"].sum())
B = 10000
conv_ad = rng.hypergeometric(n_conv, n_total - n_conv, n_ad, size=B)
null_diff = conv_ad / n_ad - (n_conv - conv_ad) / (n_total - n_ad)

# 5. Distribuição nula.
plt.figure(figsize=(9.4, 5.2))
sns.histplot(null_diff * 100, bins=45, color=LIGHT, edgecolor="white")
plt.axvline(observed * 100, color=ORANGE, lw=3, label=f"observada = {observed*100:.2f} p.p.")
plt.xlabel("Diferença: anúncio − PSA (p.p.)")
plt.ylabel("Permutações")
plt.title("Sob H₀, diferenças grandes quase não aparecem")
plt.legend()
finish("distribuicao-nula.png")

# 6. Uma permutação pequena, visual.
toy = pd.DataFrame({"resultado": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0], "grupo": ["A"]*5+["B"]*5})
toy["grupo_permutado"] = rng.permutation(toy["grupo"])
fig, axes = plt.subplots(2, 1, figsize=(10, 4.8))
for ax, col, title in zip(axes, ["grupo", "grupo_permutado"], ["Rótulos observados", "Uma permutação sob H₀"]):
    colors = toy[col].map({"A": ORANGE, "B": BLUE})
    ax.scatter(range(len(toy)), toy["resultado"], c=colors, s=180)
    ax.set_yticks([0, 1], ["não", "sim"])
    ax.set_title(title, loc="left")
    ax.set_xticks([])
finish("uma-permutacao.png")

# 7. Convergência do p-valor de Monte Carlo.
extreme = np.abs(null_diff) >= abs(observed)
check = np.arange(100, B + 1, 100)
p_running = np.cumsum(extreme)[check - 1] / check
plt.figure(figsize=(9.4, 5.2))
plt.plot(check, p_running, color=BLUE, lw=2.5)
plt.axhline((extreme.sum()+1)/(B+1), color=ORANGE, ls="--")
plt.xlabel("Permutações")
plt.ylabel("Estimativa do p-valor")
plt.title("Mais permutações estabilizam o erro de Monte Carlo")
finish("convergencia-pvalor.png")

# 8. Caudas unilateral e bilateral.
x = np.linspace(-4, 4, 500)
y = np.exp(-x*x/2)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, title in zip(axes, ["Unilateral: anúncio melhora", "Bilateral: grupos diferem"]):
    ax.plot(x, y, color=INK, lw=2.5)
    ax.set_title(title)
    ax.set_yticks([])
axes[0].fill_between(x, 0, y, where=x >= 2, color=ORANGE)
axes[1].fill_between(x, 0, y, where=np.abs(x) >= 2, color=ORANGE)
finish("uma-duas-caudas.png")

# 9. Efeito absoluto e relativo.
control, treatment = rates.iloc[0], rates.iloc[1]
absolute = treatment - control
relative = treatment / control - 1
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
axes[0].bar(["Diferença"], [100*absolute], color=BLUE)
axes[0].set_ylabel("pontos percentuais")
axes[0].set_title(f"+{100*absolute:.2f} p.p.")
axes[1].bar(["Uplift"], [100*relative], color=ORANGE)
axes[1].set_ylabel("porcentagem")
axes[1].set_title(f"+{100*relative:.1f}%")
finish("efeito-absoluto-relativo.png")

# 10. Estatísticas alternativas para total de anúncios.
ads = df[df["total ads"] <= df["total ads"].quantile(.995)]
plt.figure(figsize=(9.4, 5.2))
sns.boxplot(data=ads, x="grupo", y="total ads", showfliers=False, palette=[ORANGE, BLUE])
plt.xlabel("")
plt.ylabel("Total de exposições")
plt.title("A cauda torna a média de exposições pouco robusta")
finish("boxplot-exposicoes.png")

# 11. Diferenças por hora.
hour = df.groupby(["most ads hour", "grupo"], observed=True)["converted"].mean().mul(100).unstack()
hour["diferença"] = hour["Tratamento (anúncio)"] - hour["Controle (PSA)"]
plt.figure(figsize=(10, 5.1))
plt.axhline(0, color=INK, lw=1)
plt.bar(hour.index, hour["diferença"], color=np.where(hour["diferença"] >= 0, BLUE, ORANGE))
plt.xlabel("Hora de maior exposição")
plt.ylabel("Diferença de conversão (p.p.)")
plt.title("O efeito estimado varia entre estratos pequenos")
finish("diferenca-hora.png")

# 12. Composição por dia (checagem de balanceamento).
composition = pd.crosstab(df["most ads day"], df["test group"], normalize="columns").reindex(day_order)
(100*composition).plot.bar(figsize=(10, 5.2), color=[BLUE, ORANGE])
plt.xlabel("")
plt.ylabel("Composição do grupo (%)")
plt.title("Os dias têm composição parecida nos dois grupos")
plt.legend(["Anúncio", "PSA"], title="")
finish("balanceamento-dia.png")

# 13. Poder aproximado por tamanho amostral e efeito.
sizes = np.array([500, 1000, 2500, 5000, 10000, 25000])
base = control
effects = [0.0025, 0.005, 0.01]
from scipy.stats import norm
plt.figure(figsize=(9.4, 5.2))
for effect, color in zip(effects, [LIGHT, ORANGE, BLUE]):
    se0 = np.sqrt(2 * base * (1-base) / sizes)
    power = 1 - norm.cdf(1.96 - effect/se0) + norm.cdf(-1.96 - effect/se0)
    plt.plot(sizes, power, marker="o", lw=2.5, color=color, label=f"efeito = {100*effect:.2f} p.p.")
plt.axhline(.8, color=INK, ls="--")
plt.xscale("log")
plt.xlabel("Usuários por grupo (escala log)")
plt.ylabel("Poder aproximado")
plt.title("Efeitos pequenos exigem amostras grandes")
plt.legend()
finish("poder-tamanho.png")

print(summary)
print(f"diferença={observed:.6f}; p bilateral={(extreme.sum()+1)/(B+1):.6f}")
