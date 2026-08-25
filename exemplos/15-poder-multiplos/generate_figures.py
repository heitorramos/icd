"""Gera figuras da Aula 15 com a base Cookie Cats."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/cookie-cats/cookie_cats.csv"
DATA_URL = "https://raw.githubusercontent.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv"
OUT = ROOT / "slides/assets/aula15-poder-multiplos"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, PURPLE, GREEN, INK, LIGHT = "#0f6b78", "#d95f02", "#6a4c93", "#3a7d44", "#17324d", "#cbd5da"
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 160, "axes.titleweight": "bold"})
rng = np.random.default_rng(1414)


def finish(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close()


df = pd.read_csv(DATA if DATA.exists() else DATA_URL)
labels = {"gate_30": "Porta no nível 30", "gate_40": "Porta no nível 40"}
df["grupo"] = df["version"].map(labels)
order = [labels["gate_30"], labels["gate_40"]]

# 1. Tamanho dos grupos.
counts = df["grupo"].value_counts().reindex(order)
plt.figure(figsize=(9, 5.1))
counts.plot.bar(color=[ORANGE, BLUE])
plt.xticks(rotation=0)
plt.ylabel("Jogadores")
plt.title("Os grupos têm tamanhos muito semelhantes")
for i, v in enumerate(counts): plt.text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontweight="bold")
finish("tamanho-grupos.png")

# 2. Retenção observada.
ret = df.groupby("grupo", observed=True)[["retention_1", "retention_7"]].mean().reindex(order).mul(100)
ret.columns = ["1 dia", "7 dias"]
ret.T.plot.bar(figsize=(9.5, 5.2), color=[ORANGE, BLUE])
plt.xticks(rotation=0)
plt.ylabel("Retenção (%)")
plt.title("A porta no nível 30 retém um pouco mais")
plt.legend(title="")
finish("retencao-observada.png")

# 3. Rodadas: cauda e outlier.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
sns.histplot(df.loc[df.sum_gamerounds <= 500, "sum_gamerounds"], bins=50, color=BLUE, ax=axes[0])
axes[0].set_xlabel("Rodadas em 14 dias")
axes[0].set_title("99% dos jogadores")
sns.boxplot(data=df[df.sum_gamerounds <= df.sum_gamerounds.quantile(.999)], x="grupo", y="sum_gamerounds", showfliers=False, palette=[ORANGE, BLUE], ax=axes[1])
axes[1].set_xlabel("")
axes[1].set_ylabel("Rodadas")
axes[1].set_title("Centro por grupo")
fig.suptitle("Engajamento tem uma cauda muito longa", fontweight="bold")
finish("rodadas-distribuicao.png")

# Vetores da métrica primária.
y30 = df.loc[df.version.eq("gate_30"), "retention_7"].astype(float).to_numpy()
y40 = df.loc[df.version.eq("gate_40"), "retention_7"].astype(float).to_numpy()
obs = y30.mean() - y40.mean()
B = 10000

# 4. Permutação sob H0, eficiente via hipergeométrica.
n30, n40 = len(y30), len(y40)
success = int(y30.sum() + y40.sum())
s30 = rng.hypergeometric(success, n30+n40-success, n30, size=B)
perm = s30/n30 - (success-s30)/n40
plt.figure(figsize=(9.4, 5.2))
sns.histplot(100*perm, bins=45, color=LIGHT)
plt.axvline(100*obs, color=ORANGE, lw=3, label=f"observada = {100*obs:.2f} p.p.")
plt.xlabel("Diferença de retenção de 7 dias (p.p.)")
plt.ylabel("Permutações")
plt.title("Permutação constrói diretamente o mundo sem efeito")
plt.legend()
finish("permutacao-h0.png")

# 5. Bootstrap natural sob HA.
boot30 = rng.binomial(n30, y30.mean(), size=B)/n30
boot40 = rng.binomial(n40, y40.mean(), size=B)/n40
boot_alt = boot30-boot40
plt.figure(figsize=(9.4, 5.2))
sns.histplot(100*boot_alt, bins=45, color=BLUE)
plt.axvline(0, color=INK, ls="--")
plt.axvline(100*obs, color=ORANGE, lw=3)
plt.xlabel("Diferença bootstrap (p.p.)")
plt.ylabel("Réplicas")
plt.title("Bootstrap preserva naturalmente o efeito observado")
finish("bootstrap-ha.png")

# 6. Bootstrap recentralizado sob H0.
pooled = (y30.sum()+y40.sum())/(n30+n40)
r30 = y30 - y30.mean() + pooled
r40 = y40 - y40.mean() + pooled
# Para Bernoulli, gerar da proporção comum é a versão válida e simples da nula.
null30 = rng.binomial(n30, pooled, size=B)/n30
null40 = rng.binomial(n40, pooled, size=B)/n40
boot_null = null30-null40
plt.figure(figsize=(9.4, 5.2))
sns.histplot(100*boot_null, bins=45, color=PURPLE)
plt.axvline(100*obs, color=ORANGE, lw=3)
plt.axvline(0, color=INK, ls="--")
plt.xlabel("Diferença bootstrap recentralizada (p.p.)")
plt.ylabel("Réplicas")
plt.title("Recentrar permite usar bootstrap sob H0")
finish("bootstrap-h0-recentrado.png")

# 7. Comparação dos três mundos.
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
for ax, values, title, color in zip(axes, [perm, boot_null, boot_alt], ["Permutação: H0", "Bootstrap recentrado: H0", "Bootstrap usual: HA"], [LIGHT, PURPLE, BLUE]):
    sns.histplot(100*values, bins=35, color=color, ax=ax)
    ax.axvline(100*obs, color=ORANGE, lw=2)
    ax.set_title(title)
    ax.set_xlabel("Diferença (p.p.)")
fig.suptitle("A simulação deve corresponder à hipótese", fontweight="bold")
finish("tres-distribuicoes.png")

# 8. Poder por efeito e tamanho.
p0 = y40.mean()
sizes = np.array([500, 1000, 2500, 5000, 10000, 25000, 45000])
effects = [0.0025, 0.005, obs]
plt.figure(figsize=(9.5, 5.2))
for effect, color in zip(effects, [LIGHT, ORANGE, BLUE]):
    se0 = np.sqrt(2*p0*(1-p0)/sizes)
    power = 1-norm.cdf(1.96-effect/se0)+norm.cdf(-1.96-effect/se0)
    plt.plot(sizes, power, marker="o", lw=2.5, color=color, label=f"efeito = {100*effect:.2f} p.p.")
plt.axhline(.8, color=INK, ls="--", label="80%")
plt.xscale("log")
plt.ylim(0,1.03)
plt.xlabel("Jogadores por grupo (escala log)")
plt.ylabel("Poder aproximado")
plt.title("Poder cresce com amostra e efeito")
plt.legend()
finish("poder-efeito-amostra.png")

# 9. Alfa e poder.
alphas = np.array([.001, .005, .01, .025, .05, .10])
n = 10000
se = np.sqrt(2*p0*(1-p0)/n)
zcrit = norm.ppf(1-alphas/2)
power_alpha = 1-norm.cdf(zcrit-obs/se)+norm.cdf(-zcrit-obs/se)
plt.figure(figsize=(9.2, 5.1))
plt.plot(100*alphas, power_alpha, marker="o", color=BLUE, lw=3)
plt.xlabel("Nível de significância α (%)")
plt.ylabel("Poder")
plt.title("Reduzir falsos positivos também reduz poder")
finish("alfa-poder.png")

# 10. Muitas hipóteses nulas.
m = 100
pvals = rng.uniform(size=m)
plt.figure(figsize=(9.4, 5.1))
colors = np.where(pvals < .05, ORANGE, LIGHT)
plt.scatter(np.arange(1,m+1), pvals, c=colors, s=45)
plt.axhline(.05, color=INK, ls="--", label="α = 0,05")
plt.xlabel("Teste")
plt.ylabel("Valor-p")
plt.title("Mesmo sem efeitos, alguns valores-p ficam pequenos")
plt.legend()
finish("cem-pvalores.png")

# 11. FWER conforme m.
ms = np.arange(1,101)
fwer = 1-(1-.05)**ms
plt.figure(figsize=(9.2, 5.1))
plt.plot(ms, fwer, color=BLUE, lw=3)
plt.axhline(.95, color=ORANGE, ls="--")
plt.xlabel("Número de testes independentes")
plt.ylabel("P(ao menos um falso positivo)")
plt.title("A chance de algum falso positivo aproxima-se de 1")
finish("fwer-testes.png")

# 12. Bonferroni e BH em p-valores ilustrativos.
p = np.sort(np.r_[rng.uniform(size=44), [.0004,.001,.004,.009,.015,.03]])
m = len(p)
bh = .05*np.arange(1,m+1)/m
plt.figure(figsize=(9.4,5.2))
plt.scatter(np.arange(1,m+1), p, color=BLUE, label="valores-p ordenados")
plt.plot(np.arange(1,m+1), bh, color=GREEN, lw=2.5, label="limiares BH")
plt.axhline(.05/m, color=ORANGE, ls="--", lw=2.5, label="Bonferroni")
plt.ylim(0,.07)
plt.xlabel("Posição i")
plt.ylabel("Valor-p")
plt.title("Bonferroni e BH respondem a objetivos diferentes")
plt.legend()
finish("bonferroni-bh.png")

# 13. Winner's curse.
true_effect = .003
se = .004
estimates = rng.normal(true_effect, se, 5000)
selected = estimates[estimates > 1.96*se]
plt.figure(figsize=(9.3,5.1))
sns.histplot(100*estimates, bins=45, color=LIGHT, label="todos os experimentos")
sns.histplot(100*selected, bins=25, color=ORANGE, label="apenas significativos")
plt.axvline(100*true_effect, color=INK, lw=3, label="efeito verdadeiro")
plt.xlabel("Efeito estimado (p.p.)")
plt.ylabel("Experimentos")
plt.title("Selecionar só resultados significativos exagera o efeito")
plt.legend()
finish("winners-curse.png")

print(df.groupby("version").agg(n=("userid","size"),r1=("retention_1","mean"),r7=("retention_7","mean"),rounds=("sum_gamerounds","mean")))
print(f"obs={obs:.6f}; perm_p={(np.sum(np.abs(perm)>=abs(obs))+1)/(B+1):.6f}; boot_ci={np.quantile(boot_alt,[.025,.975])}")
