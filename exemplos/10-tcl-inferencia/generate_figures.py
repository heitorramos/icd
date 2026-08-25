"""Gera as figuras das Aulas 11 e 11 com a base Olist."""

from pathlib import Path
from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tmp" / "pdfs" / "olist_orders_dataset.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/Sangamesh2703/"
    "Brazilian-E-Commerce/master/olist_orders_dataset.csv"
)
OUT = ROOT / "slides" / "assets" / "aula10-tcl-inferencia"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0f6b78"
ORANGE = "#d95f02"
PURPLE = "#6a4c93"
INK = "#17324d"
LIGHT = "#cbd5da"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 160, "axes.titleweight": "bold"})
rng = np.random.default_rng(20260824)


def finish(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close()


def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


orders = pd.read_csv(DATA if DATA.exists() else DATA_URL)
date_cols = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
orders[date_cols] = orders[date_cols].apply(pd.to_datetime)
orders = orders.dropna(subset=date_cols).copy()
orders["delivery_days"] = (
    orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400
orders["delay_days"] = (
    orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400
orders["late"] = (orders["delay_days"] > 0).astype(int)

delivery = orders["delivery_days"].to_numpy()
late = orders["late"].to_numpy()
mu = delivery.mean()
sigma = delivery.std(ddof=0)
p_late = late.mean()


def sample_means(values, n, reps=5000):
    return np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(reps)])


# 1. População: tempo de entrega.
plt.figure(figsize=(9, 5.2))
sns.histplot(delivery, bins=np.arange(0, 61, 2), color=BLUE)
plt.axvline(mu, color=ORANGE, lw=3, label=f"média = {mu:.1f} dias")
plt.xlim(0, 60)
plt.xlabel("Tempo entre compra e entrega (dias)")
plt.ylabel("Pedidos")
plt.title("O tempo de entrega é assimétrico à direita")
plt.legend()
finish("populacao-entrega.png")

# 2. Uma amostra sobre a população.
sample = rng.choice(delivery, size=50, replace=False)
plt.figure(figsize=(9, 5.2))
sns.kdeplot(delivery, color=LIGHT, lw=4, label="população")
sns.histplot(sample, bins=12, stat="density", color=ORANGE, alpha=.45, label="amostra n = 50")
plt.axvline(mu, color=INK, lw=2, ls="--", label=f"μ = {mu:.1f}")
plt.axvline(sample.mean(), color=ORANGE, lw=3, label=f"x̄ = {sample.mean():.1f}")
plt.xlim(0, 60)
plt.xlabel("Tempo de entrega (dias)")
plt.ylabel("Densidade")
plt.title("Uma amostra não reproduz perfeitamente a população")
plt.legend()
finish("amostra-entrega.png")

# 3. Distribuições amostrais por n.
means_10 = sample_means(delivery, 10)
means_50 = sample_means(delivery, 50)
means_200 = sample_means(delivery, 200)
plt.figure(figsize=(9, 5.2))
for vals, label, color in [(means_10, "n = 10", ORANGE), (means_50, "n = 50", BLUE), (means_200, "n = 200", PURPLE)]:
    sns.kdeplot(vals, lw=3, color=color, label=label)
plt.axvline(mu, color=INK, lw=2, ls="--", label="μ")
plt.xlabel("Média amostral do tempo de entrega (dias)")
plt.ylabel("Densidade")
plt.title("Aumentar n concentra a distribuição amostral")
plt.legend()
finish("tcl-tamanhos.png")

# 4. TCL para três formatos populacionais.
fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
populations = [
    (rng.uniform(0, 1, 40000), "Uniforme"),
    (rng.exponential(1, 40000), "Assimétrica"),
    (np.r_[rng.normal(-2, .6, 20000), rng.normal(2, .6, 20000)], "Bimodal"),
]
for col, (pop, title) in enumerate(populations):
    sns.histplot(pop, bins=35, color=ORANGE, ax=axes[0, col])
    axes[0, col].set_title(title)
    axes[0, col].set_ylabel("População" if col == 0 else "")
    means = sample_means(pop, 40, 4000)
    sns.histplot(means, bins=30, color=BLUE, ax=axes[1, col])
    axes[1, col].set_ylabel("Médias, n = 40" if col == 0 else "")
fig.suptitle("Populações diferentes, médias aproximadamente normais", fontweight="bold")
finish("tcl-tres-populacoes.png")

# 5. Padronização das médias.
z_means = (means_50 - mu) / (sigma / np.sqrt(50))
x = np.linspace(-4, 4, 400)
normal_pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
plt.figure(figsize=(9, 5.2))
sns.histplot(z_means, bins=35, stat="density", color=BLUE, alpha=.6, label="simulação")
plt.plot(x, normal_pdf, color=ORANGE, lw=3, label="Normal padrão")
plt.xlabel("z = (x̄ − μ) / (σ/√n)")
plt.ylabel("Densidade")
plt.title("Padronizar revela a Normal padrão")
plt.legend()
finish("padronizacao-tcl.png")

# 5b. Áreas da Normal padrão entre desvios padrão.
x_area = np.linspace(-3.7, 3.7, 1200)
pdf_area = np.exp(-x_area**2 / 2) / np.sqrt(2 * np.pi)
fig, ax = plt.subplots(figsize=(10.5, 5.4))
ax.plot(x_area, pdf_area, color=INK, lw=3)
bands = [
    (-3, -2, PURPLE, "2,1%"),
    (-2, -1, ORANGE, "13,6%"),
    (-1, 0, BLUE, "34,1%"),
    (0, 1, BLUE, "34,1%"),
    (1, 2, ORANGE, "13,6%"),
    (2, 3, PURPLE, "2,1%"),
]
for left, right, color, label in bands:
    mask = (x_area >= left) & (x_area <= right)
    ax.fill_between(x_area[mask], 0, pdf_area[mask], color=color, alpha=.82)
    midpoint = (left + right) / 2
    label_y = .010 if abs(midpoint) > 2 else (.042 if abs(midpoint) > 1 else .075)
    ax.text(midpoint, label_y, label,
            ha="center", va="center", color="white", fontsize=15,
            fontweight="bold")
for value in range(-3, 4):
    ax.axvline(value, ymin=0, ymax=.08, color=INK, lw=1.5)
ax.annotate("68,3%", xy=(0, .41), ha="center", color=BLUE,
            fontsize=18, fontweight="bold")
ax.annotate("95,4%", xy=(0, .365), ha="center", color=ORANGE,
            fontsize=18, fontweight="bold")
ax.annotate("99,7%", xy=(0, .32), ha="center", color=PURPLE,
            fontsize=18, fontweight="bold")
ax.text(-3.25, .012, "0,1%", ha="center", color=INK, fontsize=12, fontweight="bold")
ax.text(3.25, .012, "0,1%", ha="center", color=INK, fontsize=12, fontweight="bold")
ax.set_xlim(-3.7, 3.7)
ax.set_ylim(0, .46)
ax.set_xticks(range(-3, 4))
ax.set_xticklabels(["−3σ", "−2σ", "−1σ", "μ", "+1σ", "+2σ", "+3σ"])
ax.set_yticks([])
ax.set_xlabel("Distância em relação à média")
ax.set_title("Na Normal, área representa proporção")
sns.despine(left=True)
finish("areas-normal.png")

# 6. Erro padrão por tamanho de amostra.
ns = np.array([10, 20, 50, 100, 200, 500, 1000])
se_emp = np.array([sample_means(delivery, int(n), 1800).std(ddof=1) for n in ns])
se_theory = sigma / np.sqrt(ns)
plt.figure(figsize=(9, 5.2))
plt.plot(ns, se_emp, "o-", color=ORANGE, lw=3, label="simulado")
plt.plot(ns, se_theory, "--", color=BLUE, lw=3, label="σ/√n")
plt.xscale("log")
plt.xlabel("Tamanho da amostra (escala log)")
plt.ylabel("Erro padrão (dias)")
plt.title("O erro padrão cai com a raiz de n")
plt.legend()
finish("erro-padrao-entrega.png")

# 7. Um intervalo de confiança.
n_ci = 100
sample_ci = rng.choice(delivery, n_ci, replace=False)
mean_ci = sample_ci.mean()
se_ci = sample_ci.std(ddof=1) / np.sqrt(n_ci)
lo, hi = mean_ci - 1.96 * se_ci, mean_ci + 1.96 * se_ci
plt.figure(figsize=(9, 3.8))
plt.errorbar(mean_ci, 0, xerr=1.96 * se_ci, fmt="o", color=BLUE, capsize=10, lw=4, ms=10)
plt.axvline(mu, color=ORANGE, lw=3, ls="--", label=f"μ = {mu:.1f}")
plt.yticks([])
plt.xlabel("Tempo médio de entrega (dias)")
plt.title(f"IC 95%: {lo:.1f} a {hi:.1f} dias")
plt.legend()
finish("intervalo-exemplo.png")

# 8. Cobertura de 100 intervalos.
intervals = []
for _ in range(100):
    s = rng.choice(delivery, n_ci, replace=True)
    m = s.mean()
    se = s.std(ddof=1) / np.sqrt(n_ci)
    intervals.append((m - 1.96 * se, m + 1.96 * se, m))
plt.figure(figsize=(9, 7))
for i, (lo_i, hi_i, m_i) in enumerate(intervals):
    covered = lo_i <= mu <= hi_i
    color = BLUE if covered else ORANGE
    plt.plot([lo_i, hi_i], [i, i], color=color, lw=1.7)
    plt.scatter(m_i, i, color=color, s=10)
plt.axvline(mu, color=INK, lw=2.5)
plt.xlabel("Tempo médio de entrega (dias)")
plt.ylabel("Amostras repetidas")
plt.title("95% é uma propriedade do procedimento")
finish("cobertura-intervalos.png")

# 9. Confiança versus largura.
levels = np.array([0.80, 0.90, 0.95, 0.99])
z_values = np.array([1.282, 1.645, 1.960, 2.576])
widths = 2 * z_values * sigma / np.sqrt(100)
plt.figure(figsize=(8.5, 5.2))
plt.bar(["80%", "90%", "95%", "99%"], widths, color=[LIGHT, BLUE, PURPLE, ORANGE])
plt.ylabel("Largura esperada do intervalo (dias)")
plt.xlabel("Nível de confiança")
plt.title("Mais confiança exige um intervalo mais largo")
finish("confianca-largura.png")

# 10. Margem de erro e tamanho de amostra.
ns_margin = np.arange(20, 1001, 10)
margin = 1.96 * sigma / np.sqrt(ns_margin)
plt.figure(figsize=(9, 5.2))
plt.plot(ns_margin, margin, color=BLUE, lw=3)
for target in [1.0, .5]:
    need = (1.96 * sigma / target) ** 2
    plt.axhline(target, color=ORANGE, lw=1.5, ls="--")
    plt.scatter(need, target, color=ORANGE, s=70)
plt.xlabel("Tamanho da amostra")
plt.ylabel("Margem de erro 95% (dias)")
plt.title("Planejar n começa pela precisão desejada")
finish("margem-tamanho.png")

# 11. Distribuição amostral da taxa de atraso.
prop_means = np.array([rng.choice(late, size=1200, replace=True).mean() for _ in range(5000)])
plt.figure(figsize=(9, 5.2))
sns.histplot(prop_means, bins=30, color=BLUE)
plt.axvline(p_late, color=ORANGE, lw=3, label=f"p = {p_late:.3f}")
plt.xlabel("Proporção de pedidos atrasados")
plt.ylabel("Frequência")
plt.title("A taxa de atraso também varia entre amostras")
plt.legend()
finish("distribuicao-taxa-atraso.png")

# Amostra observada usada na Aula 11.
obs = np.random.default_rng(13).choice(late, 1200, replace=False)
k_obs = int(obs.sum())
p_hat = obs.mean()
p0 = .08
se0 = np.sqrt(p0 * (1 - p0) / len(obs))
z_obs = (p_hat - p0) / se0

# 12. Distribuição nula e estatística observada.
null_props = rng.binomial(len(obs), p0, size=12000) / len(obs)
plt.figure(figsize=(9, 5.2))
sns.histplot(null_props, bins=30, color=BLUE)
plt.axvline(p_hat, color=ORANGE, lw=3, label=f"p̂ observado = {p_hat:.3f}")
plt.axvline(p0, color=INK, lw=2, ls="--", label="H0: p = 0,08")
plt.xlabel("Proporção de atrasos sob H0")
plt.ylabel("Simulações")
plt.title("A hipótese nula prevê resultados possíveis")
plt.legend()
finish("distribuicao-nula.png")

# 13. P-valor como cauda.
counts, bins = np.histogram(null_props, bins=30)
centers = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(9, 5.2))
colors = [ORANGE if c >= p_hat else BLUE for c in centers]
plt.bar(centers, counts, width=np.diff(bins), color=colors, align="center")
plt.axvline(p_hat, color=INK, lw=3)
plt.xlabel("Proporção de atrasos sob H0")
plt.ylabel("Simulações")
plt.title("O p-valor é a cauda tão extrema quanto o observado")
finish("pvalor-cauda.png")

# 14. Região de rejeição para diferentes alfas.
xz = np.linspace(-4, 4, 500)
pdf = np.exp(-xz**2 / 2) / np.sqrt(2 * np.pi)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, cutoff, alpha in [(axes[0], 1.645, "5%"), (axes[1], 2.326, "1%")]:
    ax.plot(xz, pdf, color=BLUE, lw=3)
    ax.fill_between(xz, 0, pdf, where=xz >= cutoff, color=ORANGE, alpha=.65)
    ax.axvline(cutoff, color=INK, ls="--")
    ax.set_title(f"α = {alpha}")
    ax.set_xlabel("Estatística z")
axes[0].set_ylabel("Densidade sob H0")
fig.suptitle("Reduzir α exige evidência mais extrema", fontweight="bold")
finish("regioes-rejeicao.png")

# 15. Erro tipo I por simulação.
rejections = []
running = []
for i in range(1, 5001):
    ph = rng.binomial(1200, p0) / 1200
    z = (ph - p0) / se0
    rejections.append(z > 1.645)
    running.append(np.mean(rejections))
plt.figure(figsize=(9, 5.2))
plt.plot(running, color=BLUE, lw=2)
plt.axhline(.05, color=ORANGE, lw=3, ls="--", label="α = 5%")
plt.xlabel("Número de testes simulados sob H0")
plt.ylabel("Fração de rejeições")
plt.title("Sob H0, rejeitamos cerca de alpha das vezes")
plt.legend()
finish("erro-tipo-i.png")

# 16. Curva de poder.
true_ps = np.linspace(.05, .13, 33)
critical = p0 + 1.645 * se0
power = []
for p in true_ps:
    simulated = rng.binomial(1200, p, size=5000) / 1200
    power.append(np.mean(simulated > critical))
plt.figure(figsize=(9, 5.2))
plt.plot(true_ps, power, color=BLUE, lw=3)
plt.axvline(p0, color=INK, ls="--", lw=2)
plt.axhline(.8, color=ORANGE, ls="--", lw=2, label="80% de poder")
plt.xlabel("Taxa verdadeira de atraso")
plt.ylabel("Probabilidade de rejeitar H0")
plt.title("O poder cresce quando o efeito se afasta de H0")
plt.legend()
finish("curva-poder.png")

# 17. Exata versus aproximação Normal.
n_small, p_small = 30, .5
k = np.arange(n_small + 1)
from math import comb
pmf = np.array([comb(n_small, int(i)) * p_small**i * (1-p_small)**(n_small-i) for i in k])
normal = np.exp(-((k-n_small*p_small)/np.sqrt(n_small*p_small*(1-p_small)))**2/2)
normal = normal / normal.sum()
plt.figure(figsize=(9, 5.2))
plt.vlines(k, 0, pmf, color=BLUE, lw=4, label="Binomial exata")
plt.plot(k, normal, color=ORANGE, lw=3, label="aproximação Normal")
plt.xlabel("Número de caras em 30 lançamentos")
plt.ylabel("Probabilidade")
plt.title("A Normal aproxima a Binomial no centro")
plt.legend()
finish("binomial-normal.png")

print(f"Figuras geradas em {OUT}")
print(f"N={len(orders)}; média={mu:.3f}; sigma={sigma:.3f}; atraso={p_late:.5f}")
print(f"Amostra teste: n={len(obs)}, k={k_obs}, p_hat={p_hat:.5f}, z={z_obs:.3f}, p-valor={1-normal_cdf(z_obs):.4f}")
