"""Gera as figuras da Aula 12 com a base Airbnb NYC 2019."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tmp" / "pdfs" / "AB_NYC_2019.csv"
DATA_URL = "https://raw.githubusercontent.com/YueminLi/Airbnb_NYC_2019/master/AB_NYC_2019.csv"
OUT = ROOT / "slides" / "assets" / "aula12-bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0f6b78"
ORANGE = "#d95f02"
PURPLE = "#6a4c93"
INK = "#17324d"
LIGHT = "#cbd5da"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 160, "axes.titleweight": "bold"})
rng = np.random.default_rng(1105)


def finish(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close()


raw = pd.read_csv(DATA if DATA.exists() else DATA_URL)
airbnb = raw[(raw["price"] > 0) & (raw["price"] <= raw["price"].quantile(.99))].copy()
focus = airbnb[
    airbnb["neighbourhood_group"].isin(["Manhattan", "Brooklyn"])
    & airbnb["room_type"].eq("Entire home/apt")
].copy()
sample = focus.groupby("neighbourhood_group", group_keys=False).sample(n=600, random_state=1105)
manhattan = sample.loc[sample["neighbourhood_group"].eq("Manhattan"), "price"].to_numpy()
brooklyn = sample.loc[sample["neighbourhood_group"].eq("Brooklyn"), "price"].to_numpy()
theta_hat = np.median(manhattan) - np.median(brooklyn)


def bootstrap_difference(a, b, B=5000, seed=1105):
    local_rng = np.random.default_rng(seed)
    index_a = local_rng.integers(0, len(a), size=(B, len(a)))
    index_b = local_rng.integers(0, len(b), size=(B, len(b)))
    return np.median(a[index_a], axis=1) - np.median(b[index_b], axis=1)


boot = bootstrap_difference(manhattan, brooklyn, B=3000)
ci = np.quantile(boot, [.025, .975])

# 1. Número de anúncios por distrito.
counts = raw["neighbourhood_group"].value_counts().sort_values()
plt.figure(figsize=(9, 5.2))
counts.plot.barh(color=[LIGHT, LIGHT, LIGHT, ORANGE, BLUE])
plt.xlabel("Número de anúncios")
plt.ylabel("")
plt.title("Manhattan e Brooklyn concentram 85% dos anúncios")
finish("anuncios-distrito.png")

# 2. Distribuição de preços e cauda extrema.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
sns.histplot(raw.loc[raw["price"] > 0, "price"], bins=70, color=BLUE, ax=axes[0])
axes[0].set_xlim(0, 1000)
axes[0].set_xlabel("Preço por noite (US$)")
axes[0].set_ylabel("Anúncios")
axes[0].set_title("Escala original")
sns.histplot(np.log10(raw.loc[raw["price"] > 0, "price"]), bins=45, color=ORANGE, ax=axes[1])
axes[1].set_xlabel("log10 do preço")
axes[1].set_ylabel("")
axes[1].set_title("Escala logarítmica")
fig.suptitle("Preços são fortemente assimétricos", fontweight="bold")
finish("distribuicao-precos.png")

# 3. Preços por distrito.
order = ["Bronx", "Queens", "Staten Island", "Brooklyn", "Manhattan"]
plt.figure(figsize=(10, 5.4))
sns.boxplot(data=airbnb, x="neighbourhood_group", y="price", order=order,
            showfliers=False, color=LIGHT, medianprops={"color": ORANGE, "linewidth": 3})
plt.xlabel("")
plt.ylabel("Preço por noite (US$)")
plt.title("A localização desloca toda a distribuição de preços")
finish("precos-distrito.png")

# 4. Composição por tipo de acomodação.
composition = pd.crosstab(raw["neighbourhood_group"], raw["room_type"], normalize="index")
composition = composition.loc[["Manhattan", "Brooklyn"], ["Entire home/apt", "Private room", "Shared room"]]
composition.plot.bar(stacked=True, figsize=(8.8, 5), color=[BLUE, ORANGE, PURPLE])
plt.xticks(rotation=0)
plt.xlabel("")
plt.ylabel("Proporção")
plt.title("Comparar distritos exige controlar o tipo de acomodação")
plt.legend(title="Tipo", bbox_to_anchor=(1.02, 1), loc="upper left")
finish("composicao-tipo.png")

# 5. Amostra usada na aula.
plt.figure(figsize=(9.5, 5.2))
for values, label, color in [(brooklyn, "Brooklyn", ORANGE), (manhattan, "Manhattan", BLUE)]:
    sns.ecdfplot(values, label=label, color=color, lw=3)
    plt.axvline(np.median(values), color=color, ls="--", lw=2)
plt.xlim(0, 500)
plt.xlabel("Preço por noite (US$)")
plt.ylabel("Proporção acumulada")
plt.title("A mediana compara o centro sem obedecer à cauda")
plt.legend()
finish("ecdf-amostra.png")

# 6. Uma reamostragem mostra repetições e ausências.
toy = np.array([65, 80, 90, 110, 145, 190, 240, 320])
toy_boot = np.random.default_rng(7).choice(toy, len(toy), replace=True)
fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
for ax, values, title, color in [(axes[0], toy, "Amostra observada", BLUE),
                                  (axes[1], toy_boot, "Amostra bootstrap", ORANGE)]:
    unique, multiplicity = np.unique(values, return_counts=True)
    for x, count in zip(unique, multiplicity):
        ax.scatter([x] * count, np.arange(1, count + 1), s=130, color=color)
    ax.set_yticks([])
    ax.set_title(title, loc="left")
axes[1].set_xlabel("Preço (US$)")
fig.suptitle("Reamostrar com reposição cria uma nova amostra", fontweight="bold")
finish("reamostragem-reposicao.png")

# 7. Uma única réplica bootstrap.
one_m = rng.choice(manhattan, len(manhattan), replace=True)
one_b = rng.choice(brooklyn, len(brooklyn), replace=True)
observed = [np.median(brooklyn), np.median(manhattan)]
replica = [np.median(one_b), np.median(one_m)]
xpos = np.arange(2)
plt.figure(figsize=(8.8, 4.8))
plt.plot(xpos, observed, "o-", color=BLUE, lw=3, ms=10, label="amostra observada")
plt.plot(xpos, replica, "o--", color=ORANGE, lw=3, ms=10, label="uma réplica bootstrap")
plt.xticks(xpos, ["Brooklyn", "Manhattan"])
plt.ylabel("Mediana do preço (US$)")
plt.title("Cada réplica produz uma nova diferença")
plt.legend()
finish("uma-replica.png")

# 8. Distribuição bootstrap da diferença de medianas.
plt.figure(figsize=(9.3, 5.2))
sns.histplot(boot, bins=np.arange(18, 65, 1), color=BLUE)
plt.axvline(theta_hat, color=ORANGE, lw=3, label=f"diferença observada = US$ {theta_hat:.0f}")
plt.xlabel("Mediana Manhattan − mediana Brooklyn (US$)")
plt.ylabel("Réplicas")
plt.title("As réplicas revelam a incerteza da estimativa")
plt.legend()
finish("distribuicao-bootstrap.png")

# 9. Intervalo percentil.
plt.figure(figsize=(9.3, 5.2))
sns.kdeplot(boot, fill=True, color=LIGHT, linewidth=0)
inside = (boot >= ci[0]) & (boot <= ci[1])
sns.kdeplot(boot[inside], fill=True, color=BLUE, linewidth=0)
plt.axvline(ci[0], color=ORANGE, ls="--", lw=3)
plt.axvline(ci[1], color=ORANGE, ls="--", lw=3)
plt.text(ci[0], .012, f"2,5%\nUS$ {ci[0]:.0f}", ha="right", color=ORANGE, fontweight="bold")
plt.text(ci[1], .012, f"97,5%\nUS$ {ci[1]:.0f}", ha="left", color=ORANGE, fontweight="bold")
plt.xlabel("Diferença de medianas (US$)")
plt.ylabel("Densidade")
plt.title("O intervalo percentil preserva os 95% centrais")
finish("intervalo-percentil.png")

# 10. Estabilidade conforme B aumenta.
checkpoints = np.arange(100, 3001, 100)
lower = np.array([np.quantile(boot[:b], .025) for b in checkpoints])
upper = np.array([np.quantile(boot[:b], .975) for b in checkpoints])
plt.figure(figsize=(9.4, 5.2))
plt.plot(checkpoints, lower, color=ORANGE, lw=2, label="limite inferior")
plt.plot(checkpoints, upper, color=BLUE, lw=2, label="limite superior")
plt.xlabel("Número de réplicas B")
plt.ylabel("Limite do IC (US$)")
plt.title("Mais réplicas reduzem o ruído de Monte Carlo")
plt.legend()
finish("estabilidade-b.png")

# 11. Tamanho amostral e largura do intervalo.
sizes = np.array([50, 100, 200, 400, 600])
widths = []
for n in sizes:
    widths_n = []
    for repeat in range(6):
        m = rng.choice(manhattan, n, replace=False)
        b = rng.choice(brooklyn, n, replace=False)
        values = bootstrap_difference(m, b, B=250, seed=1000 + repeat + n)
        q = np.quantile(values, [.025, .975])
        widths_n.append(q[1] - q[0])
    widths.append(widths_n)
plt.figure(figsize=(9.2, 5.2))
sns.boxplot(data=pd.DataFrame(widths, index=sizes).T, color=LIGHT,
            medianprops={"color": ORANGE, "linewidth": 3})
plt.xlabel("Observações por distrito")
plt.ylabel("Largura do IC 95% (US$)")
plt.title("Mais dados estreitam o intervalo")
finish("tamanho-largura.png")

# 12. Comparação entre métodos de intervalo.
se_boot = boot.std(ddof=1)
intervals = {
    "Percentil": ci,
    "Básico": np.array([2 * theta_hat - ci[1], 2 * theta_hat - ci[0]]),
    "Normal": np.array([theta_hat - 1.96 * se_boot, theta_hat + 1.96 * se_boot]),
}
plt.figure(figsize=(9, 4.7))
for y, (name, limits) in enumerate(intervals.items()):
    plt.plot(limits, [y, y], color=BLUE, lw=5)
    plt.scatter(theta_hat, y, color=ORANGE, s=90, zorder=3)
plt.yticks(range(3), intervals.keys())
plt.axvline(0, color=INK, ls="--")
plt.xlabel("Diferença de medianas (US$)")
plt.title("Métodos diferentes podem produzir limites diferentes")
finish("metodos-intervalo.png")

# 13. Cobertura em um experimento controlado.
population_median = np.median(focus.loc[focus["neighbourhood_group"].eq("Manhattan"), "price"])
population_median -= np.median(focus.loc[focus["neighbourhood_group"].eq("Brooklyn"), "price"])
cover = []
for i in range(20):
    m = rng.choice(focus.loc[focus["neighbourhood_group"].eq("Manhattan"), "price"], 250, replace=False)
    b = rng.choice(focus.loc[focus["neighbourhood_group"].eq("Brooklyn"), "price"], 250, replace=False)
    values = bootstrap_difference(m, b, B=300, seed=3000 + i)
    lo, hi = np.quantile(values, [.025, .975])
    cover.append((lo, hi, lo <= population_median <= hi))
plt.figure(figsize=(9, 6.3))
for i, (lo, hi, covered) in enumerate(cover):
    plt.plot([lo, hi], [i, i], color=BLUE if covered else ORANGE, lw=2)
plt.axvline(population_median, color=INK, lw=3)
plt.xlabel("Diferença de medianas (US$)")
plt.ylabel("Amostras repetidas")
plt.title("Validade significa cobertura no longo prazo")
finish("cobertura-bootstrap.png")

# 14. Listagens não são necessariamente independentes.
host_counts = sample.groupby(["neighbourhood_group", "host_id"]).size()
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
sns.histplot(host_counts, discrete=True, color=BLUE, ax=axes[0])
axes[0].set_xlim(.5, 6.5)
axes[0].set_xlabel("Anúncios do mesmo anfitrião na amostra")
axes[0].set_ylabel("Anfitriões")
axes[0].set_title("Há agrupamento por anfitrião")
naive = bootstrap_difference(manhattan, brooklyn, B=1000, seed=9)
cluster_estimates = []
for _ in range(100):
    pieces = []
    for borough in ["Manhattan", "Brooklyn"]:
        group = sample[sample["neighbourhood_group"].eq(borough)]
        hosts = group["host_id"].unique()
        chosen = rng.choice(hosts, len(hosts), replace=True)
        values = np.concatenate([group.loc[group["host_id"].eq(h), "price"].to_numpy() for h in chosen])
        pieces.append(np.median(values))
    cluster_estimates.append(pieces[0] - pieces[1])
sns.kdeplot(naive, color=BLUE, lw=3, label="por anúncio", ax=axes[1])
sns.kdeplot(cluster_estimates, color=ORANGE, lw=3, label="por anfitrião", ax=axes[1])
axes[1].set_xlabel("Diferença de medianas (US$)")
axes[1].set_ylabel("Densidade")
axes[1].set_title("A unidade de reamostragem importa")
axes[1].legend()
finish("dependencia-host.png")

print(f"linhas={len(raw)}; análise={len(airbnb)}; foco={len(focus)}")
print(f"medianas: Manhattan={np.median(manhattan):.0f}; Brooklyn={np.median(brooklyn):.0f}")
print(f"diferença={theta_hat:.0f}; IC95%=[{ci[0]:.0f}, {ci[1]:.0f}]")
