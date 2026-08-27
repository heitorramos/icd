"""Gera figuras da Aula 17: regressão linear simples por mínimos quadrados."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "exemplos/16-correlacao/data/insurance.csv"
OUT = ROOT / "slides/assets/aula17-regressao-linear"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, PURPLE, GREEN = "#0f6b78", "#d95f02", "#6a4c93", "#3a7d44"
INK, LIGHT, RED = "#17324d", "#cbd5da", "#b23a48"
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 170, "axes.titleweight": "bold"})
rng = np.random.default_rng(1717)


def finish(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close()


df = pd.read_csv(DATA)
df["fumante"] = df.smoker.map({"yes": "Fumante", "no": "Não fumante"})
x = df.age.to_numpy(float)
y = df.charges.to_numpy(float)
xbar, ybar = x.mean(), y.mean()
b1 = np.sum((x-xbar)*(y-ybar))/np.sum((x-xbar)**2)
b0 = ybar-b1*xbar
yhat = b0+b1*x
resid = y-yhat
sse = np.sum(resid**2)
sst = np.sum((y-ybar)**2)
r2 = 1-sse/sst
rmse = np.sqrt(np.mean(resid**2))

# 1. Descrição da base orientada à regressão.
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
sns.histplot(df, x="age", bins=24, color=BLUE, ax=axes[0])
sns.histplot(df, x="charges", bins=35, color=ORANGE, ax=axes[1])
sns.boxplot(df, x="fumante", y="charges", hue="fumante", palette=[ORANGE, BLUE], legend=False, ax=axes[2])
axes[0].set(xlabel="Idade", ylabel="Pessoas", title="Entrada X")
axes[1].set(xlabel="Despesas (US$)", ylabel="Pessoas", title="Resposta Y")
axes[2].set(xlabel="", ylabel="Despesas (US$)", title="Um grupo importante")
fig.suptitle("Antes do modelo: escalas, assimetria e heterogeneidade", fontweight="bold")
finish("descritiva-regressao.png")

# 2. Dispersão e média global.
plt.figure(figsize=(9.5, 5.3))
plt.scatter(x, y, color=BLUE, alpha=.28, s=24)
plt.axhline(ybar, color=ORANGE, lw=3, label=rf"média de Y = US$ {ybar:,.0f}")
plt.xlabel("Idade (anos)"); plt.ylabel("Despesas (US$)")
plt.title("Prever a média ignora a informação contida em X")
plt.legend()
finish("media-global.png")

# 3. Médias locais por faixa e linha OLS.
bins = pd.cut(df.age, bins=np.arange(17.5, 65.5, 5))
local = df.groupby(bins, observed=True).agg(x=("age", "mean"), y=("charges", "mean"), n=("charges", "size"))
grid = np.linspace(x.min(), x.max(), 100)
plt.figure(figsize=(9.5, 5.4))
plt.scatter(x, y, color=LIGHT, alpha=.35, s=20, label="pessoas")
plt.scatter(local.x, local.y, s=35+local.n, color=ORANGE, edgecolor="white", label="médias por faixa")
plt.plot(grid, b0+b1*grid, color=INK, lw=3, label="reta de mínimos quadrados")
plt.xlabel("Idade (anos)"); plt.ylabel("Despesas (US$)")
plt.title("A reta aproxima a tendência das médias condicionais")
plt.legend()
finish("medias-locais.png")

# 4. Três retas candidatas e resíduos verticais.
sample = df.sample(70, random_state=17).sort_values("age")
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharex=True, sharey=True)
slopes = [0, b1, 2*b1]
for ax, slope, color in zip(axes, slopes, [PURPLE, GREEN, RED]):
    intercept = ybar-slope*xbar
    pred = intercept+slope*sample.age
    for xx, yy, pp in zip(sample.age, sample.charges, pred):
        ax.plot([xx, xx], [pp, yy], color=LIGHT, lw=.8)
    ax.scatter(sample.age, sample.charges, color=BLUE, alpha=.55, s=18)
    ax.plot(grid, intercept+slope*grid, color=color, lw=3)
    mse = np.mean((sample.charges-pred)**2)
    ax.set_title(rf"$\beta_1={slope:.0f}$"+f"\nMSE = {mse/1e6:.1f} milhões")
    ax.set_xlabel("Idade")
axes[0].set_ylabel("Despesas (US$)")
fig.suptitle("Cada reta produz um conjunto diferente de resíduos", fontweight="bold")
finish("retas-candidatas.png")

# 5. Anatomia de um resíduo.
idx = np.argmin(np.abs(x-45) + np.abs(y-25000)/3000)
xp, yp, hp = x[idx], y[idx], yhat[idx]
plt.figure(figsize=(9.2, 5.2))
plt.scatter(x, y, color=LIGHT, alpha=.35, s=20)
plt.plot(grid, b0+b1*grid, color=BLUE, lw=3)
plt.scatter([xp], [yp], color=ORANGE, s=110, zorder=5, label=rf"observado $y_i={yp:,.0f}$")
plt.scatter([xp], [hp], color=INK, s=90, zorder=5, label=rf"previsto $\hat y_i={hp:,.0f}$")
plt.annotate("", xy=(xp, yp), xytext=(xp, hp), arrowprops={"arrowstyle": "<->", "color": RED, "lw": 3})
plt.text(xp+1, (yp+hp)/2, rf"$e_i=y_i-\hat y_i={yp-hp:,.0f}$", color=RED, va="center")
plt.xlabel("Idade"); plt.ylabel("Despesas (US$)")
plt.title("O resíduo é a distância vertical até a reta")
plt.legend()
finish("anatomia-residuo.png")

# 6. Perda em função da inclinação, mantendo a reta no centroide.
slopes_grid = np.linspace(-150, 850, 350)
loss = np.array([np.mean((y-(ybar-s*xbar+s*x))**2) for s in slopes_grid])
plt.figure(figsize=(9.2, 5.1))
plt.plot(slopes_grid, loss/1e6, color=PURPLE, lw=3)
plt.axvline(b1, color=ORANGE, ls="--", lw=3, label=rf"mínimo: $\hat\beta_1={b1:.1f}$")
plt.scatter([b1], [np.mean(resid**2)/1e6], color=ORANGE, s=100)
plt.xlabel(r"Inclinação candidata $\beta_1$")
plt.ylabel("MSE (milhões de US$²)")
plt.title("A perda quadrática é convexa e tem um único mínimo")
plt.legend()
finish("perda-inclinacao.png")

# 7. Ajuste OLS final, colorido por tabagismo.
plt.figure(figsize=(9.6, 5.4))
sns.scatterplot(df, x="age", y="charges", hue="fumante", palette=[BLUE, ORANGE], alpha=.55, s=30)
plt.plot(grid, b0+b1*grid, color=INK, lw=4, label="OLS global")
plt.scatter([xbar], [ybar], marker="X", s=150, color=GREEN, label=r"$(\bar x,\bar y)$")
plt.xlabel("Idade"); plt.ylabel("Despesas (US$)")
plt.title("A reta ajustada passa pelo centro dos dados")
plt.legend(ncol=2)
finish("ajuste-final.png")

# 8. Soma total versus resíduos do ajuste.
subset = df.sample(65, random_state=23)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharex=True, sharey=True)
for xx, yy in zip(subset.age, subset.charges):
    axes[0].plot([xx, xx], [ybar, yy], color=LIGHT, lw=1)
    axes[1].plot([xx, xx], [b0+b1*xx, yy], color=LIGHT, lw=1)
for ax in axes: ax.scatter(subset.age, subset.charges, color=BLUE, alpha=.65, s=22); ax.set_xlabel("Idade")
axes[0].axhline(ybar, color=ORANGE, lw=3); axes[0].set_title("Modelo-base: prever a média\nSST")
axes[1].plot(grid, b0+b1*grid, color=GREEN, lw=3); axes[1].set_title("Modelo linear\nSSE")
axes[0].set_ylabel("Despesas (US$)")
fig.suptitle(r"$R^2=1-\mathrm{SSE}/\mathrm{SST}$ compara dois conjuntos de erros", fontweight="bold")
finish("decomposicao-r2.png")

# 9. Resíduos contra ajustados.
plt.figure(figsize=(9.4, 5.2))
sns.scatterplot(x=yhat, y=resid, hue=df.fumante, palette=[BLUE, ORANGE], alpha=.55, s=28)
plt.axhline(0, color=INK, lw=2)
plt.xlabel("Valor ajustado (US$)"); plt.ylabel("Resíduo (US$)")
plt.title("Os resíduos revelam grupos e variância não constante")
plt.legend(title="")
finish("residuos-ajustados.png")

# 10. QQ plot e histograma dos resíduos.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(resid, bins=38, color=PURPLE, ax=axes[0])
axes[0].set(xlabel="Resíduo (US$)", ylabel="Pessoas", title="Distribuição dos resíduos")
stats.probplot(resid, dist="norm", plot=axes[1])
axes[1].get_lines()[0].set_color(BLUE); axes[1].get_lines()[1].set_color(ORANGE)
axes[1].set_title("QQ plot normal")
fig.suptitle("O modelo simples não satisfaz bem os diagnósticos", fontweight="bold")
finish("diagnostico-residuos.png")

# 11. Bootstrap pareado da inclinação.
B = 4000
boot = np.empty(B)
for j in range(B):
    ids = rng.integers(0, len(x), len(x))
    xb, yb = x[ids], y[ids]
    boot[j] = np.sum((xb-xb.mean())*(yb-yb.mean()))/np.sum((xb-xb.mean())**2)
ci = np.quantile(boot, [.025, .975])
plt.figure(figsize=(9.2, 5.0))
sns.histplot(boot, bins=42, color=BLUE)
plt.axvline(b1, color=ORANGE, lw=3, label=rf"$\hat\beta_1={b1:.1f}$")
plt.axvspan(ci[0], ci[1], color=ORANGE, alpha=.16, label=rf"IC 95% [{ci[0]:.1f}; {ci[1]:.1f}]")
plt.xlabel("Inclinação bootstrap (US$/ano)"); plt.ylabel("Réplicas")
plt.title("Reamostrar pares quantifica a incerteza da inclinação")
plt.legend()
finish("bootstrap-inclinacao.png")

# 12. Permutação sob ausência de associação.
perm = np.empty(B)
den = np.sum((x-xbar)**2)
for j in range(B):
    yp_perm = rng.permutation(y)
    perm[j] = np.sum((x-xbar)*(yp_perm-yp_perm.mean()))/den
p_perm = (1+np.sum(np.abs(perm) >= abs(b1)))/(B+1)
plt.figure(figsize=(9.2, 5.0))
sns.histplot(perm, bins=42, color=LIGHT)
plt.axvline(b1, color=ORANGE, lw=3, label=rf"observada $={b1:.1f}$")
plt.axvline(-b1, color=ORANGE, lw=2, ls="--")
plt.xlabel("Inclinação após permutar Y (US$/ano)"); plt.ylabel("Permutações")
plt.title("Permutação constrói o mundo sem associação")
plt.legend()
finish("permutacao-inclinacao.png")

# 13. Quiz visual: qual reta minimiza os quadrados?
tiny_x = np.array([1, 2, 3, 4, 5.])
tiny_y = np.array([2, 2.8, 4.2, 4.8, 7.0])
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharex=True, sharey=True)
for ax, slope, label, color in zip(axes, [.4, 1.15, 1.8], ["A", "B", "C"], [PURPLE, GREEN, RED]):
    intercept = tiny_y.mean()-slope*tiny_x.mean()
    pred = intercept+slope*tiny_x
    ax.scatter(tiny_x, tiny_y, color=BLUE, s=45)
    ax.plot(tiny_x, pred, color=color, lw=3)
    for xx, yy, pp in zip(tiny_x, tiny_y, pred): ax.plot([xx, xx], [yy, pp], color=LIGHT, lw=2)
    ax.set_title(label, fontsize=22); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Qual reta parece minimizar a soma dos resíduos ao quadrado?", fontweight="bold")
finish("quiz-minimos-quadrados.png")

print(f"b0={b0:.4f}; b1={b1:.4f}; R2={r2:.6f}; RMSE={rmse:.2f}")
print("bootstrap CI", ci, "permutation p", p_perm)
