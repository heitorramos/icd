"""Gera as figuras da Aula 16 com a base Medical Insurance Cost."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data/insurance.csv"
OUT = ROOT / "slides/assets/aula16-correlacao"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, PURPLE, GREEN = "#0f6b78", "#d95f02", "#6a4c93", "#3a7d44"
INK, LIGHT, RED = "#17324d", "#cbd5da", "#b23a48"
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 170, "axes.titleweight": "bold"})
rng = np.random.default_rng(1616)


def finish(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close()


df = pd.read_csv(DATA)
df["fumante"] = df["smoker"].map({"yes": "Fumante", "no": "Não fumante"})

# 1. Visão descritiva da base.
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
sns.histplot(df, x="age", bins=24, color=BLUE, ax=axes[0, 0])
axes[0, 0].set(xlabel="Idade (anos)", ylabel="Pessoas", title="Idade")
sns.histplot(df, x="bmi", bins=28, color=PURPLE, ax=axes[0, 1])
axes[0, 1].set(xlabel="IMC (kg/m²)", ylabel="Pessoas", title="Índice de massa corporal")
sns.histplot(df, x="charges", bins=35, color=ORANGE, ax=axes[1, 0])
axes[1, 0].set(xlabel="Despesas médicas (US$)", ylabel="Pessoas", title="Despesas têm cauda à direita")
sns.countplot(df, x="fumante", hue="fumante", palette=[BLUE, ORANGE], legend=False, ax=axes[1, 1])
axes[1, 1].set(xlabel="", ylabel="Pessoas", title="Tabagismo")
fig.suptitle("A base combina variáveis demográficas, corporais e de custo", fontweight="bold")
finish("descritiva-base.png")

# 2. Relação idade-despesas, estratificada por tabagismo.
plt.figure(figsize=(10, 5.7))
sns.scatterplot(df, x="age", y="charges", hue="fumante", palette=[BLUE, ORANGE], alpha=.65, s=45)
plt.xlabel("Idade (anos)")
plt.ylabel("Despesas médicas (US$)")
plt.title("Idade e despesas: uma nuvem, dois regimes")
plt.legend(title="")
finish("idade-despesas-fumante.png")

# 3. Quadrantes centrados nas médias, desvios e contribuições à covariância.
sample = df.sample(180, random_state=1616)
x, y = sample["age"], sample["charges"]
xbar, ybar = x.mean(), y.mean()
same = (x-x.mean())*(y-y.mean()) >= 0
fig, ax = plt.subplots(figsize=(10.2, 6.1))
ax.axvspan(x.min()-1, xbar, color=LIGHT, alpha=.10)
ax.axvspan(xbar, x.max()+1, color=ORANGE, alpha=.035)
plt.scatter(x[same], y[same], color=GREEN, alpha=.7)
plt.scatter(x[~same], y[~same], color=RED, alpha=.7)
plt.axvline(xbar, color=INK, ls="--", lw=2)
plt.axhline(ybar, color=INK, ls=":", lw=2)
ax.text(xbar+.7, y.max()*.97, r"média $\bar{x}$", color=INK,
        rotation=90, va="top", fontsize=13)
ax.text(x.min(), ybar+700, r"média $\bar{y}$", color=INK, fontsize=13)

# Sinais dos desvios e do produto em cada quadrante.
x_left = x.min()+.12*(x.max()-x.min())
x_right = xbar+.57*(x.max()-xbar)
y_low = y.min()+.12*(ybar-y.min())
y_high = ybar+.63*(y.max()-ybar)
labels = [
    (x_left, y_high, r"$x_i-\bar{x}<0$" "\n" r"$y_i-\bar{y}>0$" "\n" r"contribuição $-$", RED),
    (x_right, y_high, r"$x_i-\bar{x}>0$" "\n" r"$y_i-\bar{y}>0$" "\n" r"contribuição $+$", GREEN),
    (x_left, y_low, r"$x_i-\bar{x}<0$" "\n" r"$y_i-\bar{y}<0$" "\n" r"contribuição $+$", GREEN),
    (x_right, y_low, r"$x_i-\bar{x}>0$" "\n" r"$y_i-\bar{y}<0$" "\n" r"contribuição $-$", RED),
]
for xx, yy, text, color in labels:
    ax.text(xx, yy, text, color=color, fontsize=13, fontweight="bold",
            ha="center", va="center",
            bbox={"boxstyle": "round,pad=.28", "fc": "white", "ec": color, "alpha": .90})

# Um ponto destacado mostra geometricamente os dois desvios.
candidate = sample[(sample.age > xbar) & (sample.charges > ybar)].sort_values("charges").iloc[len(sample[(sample.age > xbar) & (sample.charges > ybar)])//2]
xp, yp = candidate["age"], candidate["charges"]
ax.scatter([xp], [yp], color=INK, s=105, zorder=6)
ax.annotate("", xy=(xp, ybar), xytext=(xbar, ybar),
            arrowprops={"arrowstyle": "<->", "color": INK, "lw": 2.2})
ax.annotate("", xy=(xp, yp), xytext=(xp, ybar),
            arrowprops={"arrowstyle": "<->", "color": INK, "lw": 2.2})
ax.text((xbar+xp)/2, ybar-1800, r"desvio $x_i-\bar{x}$", ha="center", color=INK, fontsize=13)
ax.text(xp+1.0, (ybar+yp)/2, r"desvio $y_i-\bar{y}$", va="center", color=INK, fontsize=13, rotation=90)
plt.xlabel("Idade (anos)")
plt.ylabel("Despesas médicas (US$)")
plt.title(r"Cada ponto contribui com $(x_i-\bar{x})(y_i-\bar{y})$")
finish("quadrantes-covariancia-v2.png")

# 4. Covariância depende da unidade, correlação não.
cov_usd = np.cov(df.age, df.charges, ddof=1)[0, 1]
cov_k = np.cov(df.age, df.charges/1000, ddof=1)[0, 1]
r = df.age.corr(df.charges)
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)
for ax, yy, label, cov in [(axes[0], df.charges, "US$", cov_usd), (axes[1], df.charges/1000, "milhares de US$", cov_k)]:
    ax.scatter(df.age, yy if label == "US$" else yy*1000, color=BLUE, alpha=.22, s=22)
    ax.set_xlabel("Idade")
    ax.set_title(f"cov = {cov:,.1f}\nr = {r:.3f}".replace(",", "X").replace(".", ",").replace("X", "."))
axes[0].set_ylabel("Despesas médicas (US$)")
fig.suptitle("Trocar a unidade altera a covariância, mas não a correlação", fontweight="bold")
finish("escala-cov-cor.png")

# 5. Dados padronizados e interpretação geométrica.
zx = (df.age-df.age.mean())/df.age.std(ddof=1)
zy = (df.charges-df.charges.mean())/df.charges.std(ddof=1)
plt.figure(figsize=(8.8, 5.5))
plt.scatter(zx, zy, color=BLUE, alpha=.25, s=25)
plt.axhline(0, color=INK, lw=1.5)
plt.axvline(0, color=INK, lw=1.5)
plt.xlabel(r"Idade padronizada $z_x$")
plt.ylabel(r"Despesa padronizada $z_y$")
plt.title(r"Pearson é a média dos produtos padronizados")
finish("padronizacao.png")

# 6. Matriz de correlação numérica.
numeric = df[["age", "bmi", "children", "charges"]].rename(columns={
    "age": "idade", "bmi": "IMC", "children": "filhos", "charges": "despesas"
})
plt.figure(figsize=(8.2, 6.2))
sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, square=True)
plt.title("A matriz resume relações lineares par a par")
finish("matriz-correlacao.png")

# 7. Pearson e Spearman na relação idade-despesas.
rp = pearsonr(df.age, df.charges).statistic
rs = spearmanr(df.age, df.charges).statistic
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
sns.regplot(df, x="age", y="charges", scatter_kws={"alpha": .25, "s": 20}, line_kws={"color": ORANGE}, ax=axes[0])
axes[0].set_title(f"Valores: Pearson = {rp:.2f}")
ranked = df[["age", "charges"]].rank(method="average")
sns.regplot(x=ranked.age, y=ranked.charges, scatter_kws={"alpha": .25, "s": 20}, line_kws={"color": ORANGE}, ax=axes[1])
axes[1].set_title(f"Postos: Spearman = {rs:.2f}")
for ax in axes: ax.set(xlabel="Idade", ylabel="Despesas")
fig.suptitle("Spearman mede associação monotônica pelos postos", fontweight="bold")
finish("pearson-spearman.png")

# 8. Pearson zero não implica ausência de relação.
xs = np.linspace(-3, 3, 250)
ys = xs**2 + rng.normal(0, .45, len(xs))
rp_curve = pearsonr(xs, ys).statistic
plt.figure(figsize=(8.8, 5.3))
plt.scatter(xs, ys, color=PURPLE, alpha=.65)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Uma relação forte pode ter Pearson ≈ {rp_curve:.2f}")
finish("nao-linear-zero.png")

# 9. Sensibilidade a um ponto influente.
base = df.loc[df.smoker.eq("no"), ["bmi", "charges"]].sample(110, random_state=17)
extra = pd.DataFrame({"bmi": [54], "charges": [63000]})
r0 = base.corr().iloc[0, 1]
r1 = pd.concat([base, extra]).corr().iloc[0, 1]
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
axes[0].scatter(base.bmi, base.charges, color=BLUE, alpha=.6)
axes[0].set_title(f"Sem ponto: r = {r0:.2f}")
axes[1].scatter(base.bmi, base.charges, color=BLUE, alpha=.6)
axes[1].scatter(extra.bmi, extra.charges, color=ORANGE, s=130, marker="X")
axes[1].set_title(f"Com um ponto: r = {r1:.2f}")
for ax in axes: ax.set(xlabel="IMC", ylabel="Despesas (US$)")
fig.suptitle("Pearson pode mudar muito com um ponto influente", fontweight="bold")
finish("ponto-influente.png")

# 10. Condicionamento por tabagismo.
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), sharex=True, sharey=True)
for ax, (smoker, group), color in zip(axes, df.groupby("smoker"), [BLUE, ORANGE]):
    sns.regplot(group, x="bmi", y="charges", scatter_kws={"alpha": .45, "s": 25}, line_kws={"color": INK}, color=color, ax=ax)
    label = "Fumantes" if smoker == "yes" else "Não fumantes"
    rr = group.bmi.corr(group.charges)
    ax.set_title(f"{label}: r = {rr:.2f}")
    ax.set(xlabel="IMC", ylabel="Despesas (US$)")
fig.suptitle("A associação IMC-despesas depende do estrato", fontweight="bold")
finish("correlacao-condicional.png")

# 11. Restrição de amplitude.
all_r = df.age.corr(df.charges)
restricted = df[df.age.between(35, 45)]
res_r = restricted.age.corr(restricted.charges)
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
axes[0].scatter(df.age, df.charges, color=BLUE, alpha=.25, s=22)
axes[0].set_title(f"18–64 anos: r = {all_r:.2f}")
axes[1].scatter(restricted.age, restricted.charges, color=PURPLE, alpha=.45, s=28)
axes[1].set_title(f"35–45 anos: r = {res_r:.2f}")
for ax in axes: ax.set(xlabel="Idade", ylabel="Despesas (US$)")
fig.suptitle("Restringir a faixa observada altera a correlação", fontweight="bold")
finish("restricao-amplitude.png")

# 12. Incerteza por bootstrap para r(idade, despesas).
B = 4000
boot = np.empty(B)
values = df[["age", "charges"]].to_numpy()
for b in range(B):
    draw = values[rng.integers(0, len(values), len(values))]
    boot[b] = np.corrcoef(draw[:, 0], draw[:, 1])[0, 1]
ci = np.quantile(boot, [.025, .975])
plt.figure(figsize=(9.0, 5.0))
sns.histplot(boot, bins=40, color=BLUE)
plt.axvline(rp, color=ORANGE, lw=3, label=f"r = {rp:.3f}")
plt.axvspan(ci[0], ci[1], color=ORANGE, alpha=.17, label=f"IC 95% [{ci[0]:.3f}; {ci[1]:.3f}]")
plt.xlabel("Correlação de Pearson bootstrap")
plt.ylabel("Réplicas")
plt.title("Correlação amostral também tem incerteza")
plt.legend()
finish("bootstrap-correlacao.png")

# 13. Quiz visual: associe cada nuvem à correlação de Pearson.
targets = [("A", 0.00), ("B", 0.90), ("C", -0.55), ("D", 0.25)]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharex=True, sharey=True)
for ax, (label, rho) in zip(axes, targets):
    qx = rng.normal(size=180)
    qnoise = rng.normal(size=180)
    qy = rho*qx + np.sqrt(1-rho**2)*qnoise
    # Remove a pequena correlação amostral indesejada no painel de rho zero.
    if rho == 0:
        qy = qy - np.cov(qx, qy, ddof=0)[0, 1]/np.var(qx)*qx
    ax.scatter(qx, qy, color=BLUE, alpha=.50, s=18)
    ax.set_title(label, fontsize=24)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
fig.suptitle("Qual correlação de Pearson corresponde a cada nuvem?", fontweight="bold")
finish("quiz-correlacoes.png")

# 14. Quiz visual: relação monotônica, mas não linear.
qx = np.linspace(1, 100, 220)
qy = np.log(qx) + rng.normal(0, .045, len(qx))
quiz_rp = pearsonr(qx, qy).statistic
quiz_rs = spearmanr(qx, qy).statistic
plt.figure(figsize=(9.2, 4.6))
plt.scatter(qx, qy, color=PURPLE, alpha=.62, s=26)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("A relação preserva a ordem, mas não segue uma reta")
finish("quiz-pearson-spearman.png")

print(df.describe(include="all"))
print("Pearson idade-despesas:", rp)
print("Spearman idade-despesas:", rs)
print("IC bootstrap:", ci)
print("Quiz Pearson/Spearman:", quiz_rp, quiz_rs)
print(df.groupby("smoker")[["age", "bmi", "children", "charges"]].corr().loc[(slice(None), "charges"), :])
