from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
OUT = ROOT / "slides/assets/aula05-visualizacao"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY = "#176b87", "#d97727", "#3a7d44", "#b23b3b", "#737b83"
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold"})

penguins = pd.read_csv(DATA / "penguins.csv").rename(columns={
    "culmen_length_mm": "bill_length_mm", "culmen_depth_mm": "bill_depth_mm"
})
penguins["species"] = penguins["species"].str.split().str[0]
penguins["sex"] = penguins["sex"].replace(".", pd.NA)
datasaurus = pd.read_csv(DATA / "datasaurus.tsv", sep="\t")
olist_orders = pd.read_csv(ROOT / "exemplos/04-eda-olist/data/orders.csv",
                           parse_dates=["order_purchase_timestamp"])

def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close()

plt.figure(figsize=(9.5, 5))
sns.countplot(data=penguins, y="species", order=penguins.species.value_counts().index,
              hue="species", palette=[BLUE, ORANGE, GREEN], legend=False)
plt.title("Três espécies, frequências diferentes"); plt.xlabel("pinguins"); plt.ylabel("")
save("especies-barras.png")

fig, ax = plt.subplots(figsize=(10, 5.4))
counts = penguins.species.value_counts().sort_values()
counts.plot.barh(ax=ax, color=BLUE)
ax.set_title("Pinguins observados por espécie", loc="left")
ax.set_xlabel("número de observações"); ax.set_ylabel("")
ax.annotate("título", xy=(.02, 1.02), xycoords="axes fraction", xytext=(.28, 1.15),
            textcoords="axes fraction", arrowprops={"arrowstyle":"->","color":RED}, color=RED)
ax.annotate("área de dados", xy=(.62, .55), xycoords="axes fraction", xytext=(.72, .82),
            textcoords="axes fraction", arrowprops={"arrowstyle":"->","color":RED}, color=RED)
ax.annotate("rótulo do eixo", xy=(.55, -.10), xycoords="axes fraction", xytext=(.75, -.20),
            textcoords="axes fraction", arrowprops={"arrowstyle":"->","color":RED}, color=RED)
save("anatomia-grafico.png")

monthly=(olist_orders.set_index("order_purchase_timestamp").loc["2017"].resample("MS").size())
plt.figure(figsize=(10, 5.1))
plt.plot(monthly.index,monthly.values,color=BLUE,linewidth=2.8,marker="o")
plt.title("Pedidos da Olist ao longo de 2017"); plt.xlabel(""); plt.ylabel("pedidos na amostra")
save("linhas-tempo.png")

fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
counts.plot.barh(ax=axes[0],color=GRAY); axes[0].set_title("Padrão"); axes[0].set_xlabel(""); axes[0].set_ylabel("")
counts.plot.barh(ax=axes[1],color=["#b7c4ca","#b7c4ca",ORANGE]); axes[1].set_title("Título, rótulo e destaque"); axes[1].set_xlabel("pinguins observados"); axes[1].set_ylabel("")
fig.suptitle("Customização deve esclarecer a comparação",fontweight="bold")
save("customizacao-grafico.png")

plt.figure(figsize=(9.5, 5))
sns.histplot(data=penguins, x="body_mass_g", hue="species", bins=24,
             element="step", stat="density", common_norm=False, alpha=.20)
plt.title("Massa corporal varia entre espécies"); plt.xlabel("massa corporal (g)"); plt.ylabel("densidade")
save("massa-histograma.png")

# KDE: construção manual e papel da largura de banda.
x_obs = np.array([3200, 3500, 3800, 4300, 4750])
x_grid = np.linspace(2600, 5400, 700)

def gaussian_kde_manual(grid, observations, bandwidth):
    u = (grid[:, None] - observations[None, :]) / bandwidth
    kernels = np.exp(-0.5 * u**2) / (np.sqrt(2 * np.pi) * bandwidth)
    return kernels, kernels.mean(axis=1)

kernels, density = gaussian_kde_manual(x_grid, x_obs, 260)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True)
for curve in kernels.T:
    axes[0].plot(x_grid, curve, color=BLUE, alpha=.72, linewidth=1.8)
axes[0].plot(x_obs, np.zeros_like(x_obs), "|", color=RED, markersize=16, markeredgewidth=2)
axes[0].set_title("1. Um kernel por observação")
for curve in (kernels / len(x_obs)).T:
    axes[1].plot(x_grid, curve, color=BLUE, alpha=.72, linewidth=1.8)
axes[1].set_title(r"2. Cada área vale $1/n$")
axes[2].fill_between(x_grid, density, color=BLUE, alpha=.22)
axes[2].plot(x_grid, density, color=BLUE, linewidth=2.8)
axes[2].plot(x_obs, np.zeros_like(x_obs), "|", color=RED, markersize=16, markeredgewidth=2)
axes[2].set_title("3. Somamos os kernels")
for ax in axes:
    ax.set_xlabel("massa corporal (g)")
    ax.set_yticks([])
fig.suptitle("KDE transforma observações em uma densidade suave", fontweight="bold")
save("kde-construcao.png")

mass = penguins["body_mass_g"].dropna()
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True)
for ax, adjust, label in zip(axes, [.25, 1, 2.5], ["pequena", "intermediária", "grande"]):
    sns.kdeplot(x=mass, bw_adjust=adjust, fill=True, color=BLUE, ax=ax)
    sns.rugplot(x=mass, color=GRAY, alpha=.35, height=.045, ax=ax)
    ax.set_title(f"Largura {label}")
    ax.set_xlabel("massa corporal (g)")
axes[0].set_ylabel("densidade")
fig.suptitle("A largura de banda controla quanto detalhe permanece", fontweight="bold")
save("kde-bandwidth.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for ax, bins in zip(axes, [6, 15, 35]):
    sns.histplot(penguins["body_mass_g"].dropna(), bins=bins, color=BLUE, ax=ax)
    ax.set_title(f"{bins} bins"); ax.set_xlabel("massa (g)")
fig.suptitle("O histograma muda quando os bins mudam", fontweight="bold")
save("bins-comparacao.png")

plt.figure(figsize=(9.5, 5.2))
sns.ecdfplot(data=penguins, x="body_mass_g", hue="species",
             palette=[BLUE, ORANGE, GREEN], linewidth=2.5)
plt.axhline(.5, color=GRAY, linestyle="--", linewidth=1.4, alpha=.8)
plt.title("A ECDF compara toda a distribuição sem escolher bins")
plt.xlabel("massa corporal (g)")
plt.ylabel("proporção acumulada")
save("massa-ecdf.png")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
sns.boxplot(data=penguins, x="species", y="body_mass_g", hue="species",
            palette=[BLUE, ORANGE, GREEN], legend=False, ax=axes[0])
axes[0].set_title("Boxplot"); axes[0].set_xlabel(""); axes[0].set_ylabel("massa (g)")
sns.violinplot(data=penguins, x="species", y="body_mass_g", hue="species",
               palette=[BLUE, ORANGE, GREEN], legend=False, inner="quart", ax=axes[1])
axes[1].set_title("Violinplot"); axes[1].set_xlabel(""); axes[1].set_ylabel("")
save("box-violin.png")

plt.figure(figsize=(9, 5.4))
sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species",
                style="species", palette=[BLUE, ORANGE, GREEN], s=75)
plt.title("Posição revela grupos que resumos ocultam")
plt.xlabel("comprimento do bico (mm)"); plt.ylabel("profundidade do bico (mm)")
save("bicos-dispersao.png")

# Perceptual channels: the same 40 vs. 60 comparison.
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
axes[0, 0].scatter([0, 1], [40, 60], s=90, color=[BLUE, ORANGE])
axes[0, 0].set(ylim=(0, 70), xticks=[0, 1], xticklabels=["A", "B"], title="Posição em escala comum")
axes[0, 1].bar(["A", "B"], [40, 60], color=[BLUE, ORANGE])
axes[0, 1].set(ylim=(0, 70), title="Comprimento")
axes[1, 0].pie([40, 60], colors=[BLUE, ORANGE], startangle=90,
               wedgeprops={"edgecolor": "white"})
axes[1, 0].set_title("Ângulo")
axes[1, 1].scatter([0, 1], [0, 0], s=np.array([40, 60]) * 28,
                   color=[BLUE, ORANGE], alpha=.85)
axes[1, 1].set(xlim=(-.55, 1.55), ylim=(-.65, .65), xticks=[0, 1], xticklabels=["A", "B"], yticks=[], title="Área")
fig.suptitle("A mesma diferença exige esforços perceptivos distintos", fontweight="bold")
save("canais-perceptivos.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].barh(["1 unidade", "7 unidades"], [1, 7], color=[BLUE, ORANGE])
axes[0].set(xlim=(0, 7.5), title="Comprimento: razão visível", xlabel="valor")
axes[1].scatter([0, 1.5], [0, 0], s=np.array([1, 7]) * 900,
                color=[BLUE, ORANGE], alpha=.85)
axes[1].set(xlim=(-.6, 2.4), ylim=(-1.2, 1.2), xticks=[0, 1.5],
            xticklabels=["1 unidade", "7 unidades"], yticks=[], title="Área: razão difícil")
fig.suptitle("Os dois pares representam 1 e 7", fontweight="bold")
save("comprimento-area.png")

parts = pd.Series([32, 29, 24, 15], index=["A", "B", "C", "D"])
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].pie(parts, labels=parts.index, autopct="%1.0f%%", startangle=90,
            colors=[BLUE, ORANGE, GREEN, GRAY])
axes[0].set_title("Setores: compare ângulos")
parts.sort_values().plot.barh(ax=axes[1], color=[GRAY, GREEN, ORANGE, BLUE])
axes[1].set(xlim=(0, 35), title="Barras: compare comprimentos", xlabel="percentual", ylabel="")
fig.suptitle("Diferenças próximas aparecem melhor em barras", fontweight="bold")
save("angulos-barras.png")

markers = {"Adelie": "o", "Chinstrap": "s", "Gentoo": "^"}
plt.figure(figsize=(9.5, 5.2))
for species, marker in markers.items():
    d = penguins[penguins["species"] == species]
    plt.scatter(d["bill_length_mm"], d["bill_depth_mm"], marker=marker,
                facecolors="none", edgecolors="#263746", s=62, linewidths=1.25, label=species)
plt.title("Forma identifica categorias mesmo sem cor")
plt.xlabel("comprimento do bico (mm)"); plt.ylabel("profundidade do bico (mm)")
plt.legend(title="espécie", frameon=False)
save("formas-categorias.png")

gradient = np.linspace(0, 1, 256).reshape(1, -1)
fig, axes = plt.subplots(3, 1, figsize=(10, 4.6))
for ax, cmap, label in zip(axes, ["viridis", "gray", "jet"],
                           ["Viridis: luminância aproximadamente ordenada",
                            "Cinza: ordem explícita",
                            "Jet: mudanças artificiais de destaque"]):
    ax.imshow(gradient, aspect="auto", cmap=cmap)
    ax.set(xticks=[], yticks=[], title=label)
fig.suptitle("Uma paleta sequencial deve preservar a ordem", fontweight="bold")
save("luminancia-paletas.png")

two = penguins[penguins["species"].isin(["Adelie", "Gentoo"])]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharex=True, sharey=True)
for species, color in {"Adelie": "#d62728", "Gentoo": "#2ca02c"}.items():
    d = two[two.species == species]
    axes[0].scatter(d.bill_length_mm, d.bill_depth_mm, c=color, s=42, label=species, alpha=.75)
axes[0].set_title("Somente vermelho e verde")
for species, color, marker in [("Adelie", BLUE, "o"), ("Gentoo", ORANGE, "^")]:
    d = two[two.species == species]
    axes[1].scatter(d.bill_length_mm, d.bill_depth_mm, c=color, marker=marker,
                    s=48, label=species, alpha=.8)
axes[1].set_title("Cor acessível + forma")
for ax in axes:
    ax.set_xlabel("comprimento do bico (mm)"); ax.legend(frameon=False)
axes[0].set_ylabel("profundidade do bico (mm)")
fig.suptitle("Não dependa de um único canal visual", fontweight="bold")
save("cor-acessibilidade.png")

# Visual checks, conditioning, and limits of association.
with_error = penguins[["species", "body_mass_g"]].dropna().copy()
with_error.loc[len(with_error)] = ["Adelie", 40000]
plt.figure(figsize=(9.5, 5.1))
plt.scatter(np.arange(len(with_error)), with_error.body_mass_g,
            color=np.where(with_error.body_mass_g > 10000, RED, "#b7c4ca"), s=34)
plt.annotate("40.000 g: provável erro", (len(with_error)-1, 40000),
             xytext=(-180, -25), textcoords="offset points",
             arrowprops={"arrowstyle": "->", "color": RED}, color=RED)
plt.title("O gráfico transforma um erro de digitação em um sinal visível")
plt.xlabel("linha da base"); plt.ylabel("massa corporal (g)")
save("erro-impossivel.png")

counts = penguins.groupby(["species", "island"]).size().rename("n").reset_index()
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
for col, species in enumerate(["Adelie", "Chinstrap", "Gentoo"]):
    d = counts[counts.species == species]
    axes[0, col].bar(d.island, d.n, color=BLUE)
    axes[0, col].set_title(species); axes[0, col].tick_params(axis="x", rotation=30)
    axes[0, col].set_ylim(0, max(d.n) * 1.08)
    axes[1, col].bar(d.island, d.n, color=ORANGE)
    axes[1, col].tick_params(axis="x", rotation=30); axes[1, col].set_ylim(0, counts.n.max() * 1.08)
axes[0, 0].set_ylabel("escala livre"); axes[1, 0].set_ylabel("escala comum")
fig.suptitle("Escalas livres ocultam diferenças de magnitude", fontweight="bold")
save("facetas-escalas.png")

g = sns.FacetGrid(penguins, col="species", height=3.0, aspect=1.05, sharex=True, sharey=True)
g.map_dataframe(sns.regplot, x="flipper_length_mm", y="body_mass_g",
                scatter_kws={"s": 25, "alpha": .55, "color": BLUE},
                line_kws={"color": ORANGE})
g.set_titles("{col_name}"); g.set_axis_labels("nadadeira (mm)", "massa (g)")
g.figure.suptitle("Condicionar mostra se a relação persiste dentro dos grupos", y=1.04, fontweight="bold")
g.savefig(OUT / "facetas-relacao.png", bbox_inches="tight", facecolor="white")
plt.close(g.figure)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharex=True, sharey=True)
sns.regplot(data=penguins, x="flipper_length_mm", y="body_mass_g",
            scatter_kws={"s": 24, "alpha": .45, "color": GRAY},
            line_kws={"color": RED}, ax=axes[0])
axes[0].set_title("Associação agregada")
for species, color in zip(["Adelie", "Chinstrap", "Gentoo"], [BLUE, ORANGE, GREEN]):
    d = penguins[penguins.species == species]
    sns.regplot(data=d, x="flipper_length_mm", y="body_mass_g", ax=axes[1],
                scatter_kws={"s": 20, "alpha": .35, "color": color},
                line_kws={"color": color}, label=species)
axes[1].set_title("Relações dentro das espécies"); axes[1].legend(frameon=False)
for ax in axes: ax.set_xlabel("nadadeira (mm)")
axes[0].set_ylabel("massa (g)"); axes[1].set_ylabel("")
fig.suptitle("Associação muda quando observamos grupos relevantes", fontweight="bold")
save("associacao-grupos.png")

plt.figure(figsize=(9, 5.4))
sns.scatterplot(data=penguins, x="flipper_length_mm", y="body_mass_g", hue="species",
                palette=[BLUE, ORANGE, GREEN], s=75)
plt.title("Massa e nadadeira crescem juntas"); plt.xlabel("nadadeira (mm)"); plt.ylabel("massa (g)")
save("massa-nadadeira.png")

summ = (datasaurus.groupby("dataset").agg(mean_x=("x","mean"), mean_y=("y","mean"),
          var_x=("x","var"), var_y=("y","var"), corr=("x", lambda s: s.corr(datasaurus.loc[s.index,"y"]))).round(2))
summ.to_csv(OUT / "datasaurus-resumo.csv")

g = sns.FacetGrid(datasaurus, col="dataset", col_wrap=4, height=2.0, aspect=1.05)
g.map_dataframe(sns.scatterplot, x="x", y="y", color=BLUE, s=11)
g.set_titles("{col_name}"); g.set_axis_labels("", "")
g.figure.suptitle("Quase os mesmos resumos; formas radicalmente diferentes", y=1.02, fontweight="bold")
g.savefig(OUT / "datasaurus-grade.png", bbox_inches="tight", facecolor="white")
plt.close(g.figure)

frames=[]
order=["dino","circle","star","x_shape","bullseye","slant_up","slant_down","h_lines","v_lines","dots","wide_lines","high_lines","away"]
for name in order:
    d=datasaurus.query("dataset == @name")
    fig,ax=plt.subplots(figsize=(6.4,5.4),dpi=110)
    ax.scatter(d.x,d.y,s=26,color=BLUE,alpha=.85)
    ax.set(xlim=(0,100),ylim=(0,100),xlabel="x",ylabel="y",title=f"Datasaurus: {name}")
    ax.text(.02,.02,f"médias = ({d.x.mean():.2f}, {d.y.mean():.2f})\nvariâncias = ({d.x.var():.2f}, {d.y.var():.2f})",
            transform=ax.transAxes,fontsize=10,color=GRAY)
    fig.tight_layout(); fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
    plt.close(fig)
frames[0].save(OUT/"datasaurus-animado.gif",save_all=True,append_images=frames[1:],duration=1100,loop=0,optimize=True)

print(penguins.shape, penguins.species.value_counts().to_dict(), summ.head())
