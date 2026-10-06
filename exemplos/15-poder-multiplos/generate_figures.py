"""Gera figuras da Aula 15 com a base Cookie Cats."""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, norm

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

# Exemplo didático: randomização estratificada por plataforma.
# Em cada estrato, as cinco primeiras observações pertencem ao gate 30.
strata = {
    "Android": np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0]),
    "iOS": np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0]),
}
observed_stratified = np.mean([
    values[:5].mean() - values[5:].mean()
    for values in strata.values()
])

stratified_null = []
assignments = list(combinations(range(10), 5))
for first in assignments:
    mask_first = np.zeros(10, dtype=bool)
    mask_first[list(first)] = True
    delta_first = (
        strata["Android"][mask_first].mean()
        - strata["Android"][~mask_first].mean()
    )
    for second in assignments:
        mask_second = np.zeros(10, dtype=bool)
        mask_second[list(second)] = True
        delta_second = (
            strata["iOS"][mask_second].mean()
            - strata["iOS"][~mask_second].mean()
        )
        stratified_null.append((delta_first + delta_second) / 2)

stratified_null = np.asarray(stratified_null)
stratified_p = np.mean(
    np.abs(stratified_null) >= abs(observed_stratified) - 1e-12
)
values, frequencies = np.unique(np.round(stratified_null, 10), return_counts=True)

plt.figure(figsize=(9.6, 5.3))
tail = np.abs(values) >= abs(observed_stratified) - 1e-12
plt.bar(
    values,
    100 * frequencies / frequencies.sum(),
    width=0.16,
    color=np.where(tail, ORANGE, LIGHT),
    edgecolor="white",
)
plt.axvline(observed_stratified, color=INK, lw=3,
            label=f"observada = {observed_stratified:.2f}")
plt.xlabel("Diferença estratificada de retenção")
plt.ylabel("Permutações (%)")
plt.title("Embaralhar dentro dos estratos constrói o mundo nulo")
plt.text(-0.86, 20, f"valor-p bilateral = {stratified_p:.3f}",
         color=INK, fontweight="bold")
plt.legend(loc="upper right")
finish("permutacao-estratificada.png")

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

# Distribuições conjuntas usadas para estimar poder.
critical = np.quantile(np.abs(perm), .95)
estimated_power = np.mean(np.abs(boot_alt) >= critical)
grid = np.linspace(-1.25, 1.75, 700)
null_density = gaussian_kde(100 * perm)(grid)
alt_density = gaussian_kde(100 * boot_alt)(grid)

plt.figure(figsize=(10.2, 5.5))
plt.plot(grid, null_density, color=BLUE, lw=3, label=r"$H_0$: permutação")
plt.plot(grid, alt_density, color=ORANGE, lw=3, label=r"$H_A$: bootstrap")
rejection = np.abs(grid) >= 100 * critical
plt.fill_between(
    grid, 0, alt_density,
    where=rejection,
    color=ORANGE,
    alpha=.32,
    label=f"poder estimado = {estimated_power:.1%}",
)
plt.axvline(-100 * critical, color=INK, ls="--", lw=2)
plt.axvline(100 * critical, color=INK, ls="--", lw=2,
            label=f"limites críticos = ±{100 * critical:.2f} p.p.")
plt.axvline(100 * obs, color=PURPLE, lw=3,
            label=f"efeito sob HA = {100 * obs:.2f} p.p.")
plt.xlabel("Diferença de retenção em 7 dias (p.p.)")
plt.ylabel("Densidade")
plt.title("Poder é a área da alternativa que ultrapassa o limite crítico")
plt.legend(frameon=True, loc="upper left")
finish("poder-permutacao-bootstrap.png")

# Comparação visual dos fatores que modificam o poder.
scenario_rng = np.random.default_rng(1515)
scenario_reps = 20000
pooled_rate = (y30.sum() + y40.sum()) / (n30 + n40)


def binomial_differences(n_a, n_b, p_a, p_b):
    """Simula diferenças de proporções em experimentos independentes."""
    return (
        scenario_rng.binomial(n_a, p_a, size=scenario_reps) / n_a
        - scenario_rng.binomial(n_b, p_b, size=scenario_reps) / n_b
    )


# Referência: distribuições já construídas por permutação e bootstrap.
reference_null = perm
reference_alt = boot_alt
reference_critical = critical

# Cenário 2: 50% mais jogadores em cada grupo, mantendo as taxas observadas.
large_n30 = round(1.5 * n30)
large_n40 = round(1.5 * n40)
large_null = binomial_differences(
    large_n30, large_n40, pooled_rate, pooled_rate
)
large_alt = binomial_differences(
    large_n30, large_n40, y30.mean(), y40.mean()
)
large_critical = np.quantile(np.abs(large_null), .95)

# Cenário 3: efeito de 1,00 p.p., mantendo tamanho e variabilidade.
larger_effect = .010
effect_null = binomial_differences(n30, n40, pooled_rate, pooled_rate)
effect_alt = binomial_differences(
    n30, n40, y40.mean() + larger_effect, y40.mean()
)
effect_critical = np.quantile(np.abs(effect_null), .95)

# Cenário 4: alfa de 1%, mantendo amostra e efeito de referência.
strict_critical = np.quantile(np.abs(reference_null), .99)

power_scenarios = [
    (
        reference_null,
        reference_alt,
        reference_critical,
        obs,
        "Referência\n$\\alpha=5\\%$",
    ),
    (
        large_null,
        large_alt,
        large_critical,
        obs,
        "50% mais jogadores\nmesmo efeito",
    ),
    (
        effect_null,
        effect_alt,
        effect_critical,
        larger_effect,
        "Efeito maior\n$\\Delta=1{,}00$ p.p.",
    ),
    (
        reference_null,
        reference_alt,
        strict_critical,
        obs,
        "Teste mais rigoroso\n$\\alpha=1\\%$",
    ),
]

power_grid = np.linspace(-1.4, 2.0, 700)
fig, axes = plt.subplots(
    2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True
)

for index, (ax, scenario) in enumerate(zip(axes.flat, power_scenarios)):
    null_values, alt_values, cutoff, effect, title = scenario
    null_curve = gaussian_kde(100 * null_values)(power_grid)
    alt_curve = gaussian_kde(100 * alt_values)(power_grid)
    reject = np.abs(power_grid) >= 100 * cutoff
    scenario_power = np.mean(np.abs(alt_values) >= cutoff)

    ax.plot(
        power_grid,
        null_curve,
        color=BLUE,
        lw=2.5,
        label=r"$H_0$",
    )
    ax.plot(
        power_grid,
        alt_curve,
        color=ORANGE,
        lw=2.5,
        label=r"$H_A$",
    )
    ax.fill_between(
        power_grid,
        0,
        alt_curve,
        where=reject,
        color=ORANGE,
        alpha=.30,
        label="área do poder",
    )
    ax.axvline(
        -100 * cutoff,
        color=INK,
        ls="--",
        lw=1.6,
        label="limites críticos",
    )
    ax.axvline(100 * cutoff, color=INK, ls="--", lw=1.6)
    ax.axvline(
        100 * effect,
        color=PURPLE,
        lw=2.2,
        label="efeito sob $H_A$",
    )
    ax.set_title(title, fontsize=15)
    ax.text(
        .97,
        .92,
        f"poder = {scenario_power:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        fontweight="bold",
        fontsize=12,
    )
    if index % 2 == 0:
        ax.set_ylabel("Densidade")

handles, legend_labels = axes.flat[0].get_legend_handles_labels()
fig.supxlabel("Diferença de retenção em 7 dias (p.p.)", y=.075)
fig.legend(
    handles,
    legend_labels,
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(.5, .005),
)
fig.tight_layout(rect=(0, .13, 1, 1), h_pad=2.1, w_pad=1.2)
fig.savefig(
    OUT / "poder-quatro-cenarios.png",
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

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
print(f"stratified_p={stratified_p:.6f}; critical={critical:.6f}; power={estimated_power:.6f}")
