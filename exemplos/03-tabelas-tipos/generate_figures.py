from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data" / "AB_NYC_2019.csv"
OUT = ROOT / "slides" / "assets" / "aula03-airbnb"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
sns.set_theme(style="whitegrid", font_scale=1.05)
blue, orange, green = "#0f6b78", "#d95f02", "#16826c"

# Visão espacial: uma linha = um anúncio.
fig, ax = plt.subplots(figsize=(9.5, 6.2))
for borough, part in df.groupby("neighbourhood_group"):
    ax.scatter(part.longitude, part.latitude, s=5, alpha=.32, label=borough)
ax.set(xlabel="longitude", ylabel="latitude", title="48.895 anúncios formam o mapa de Nova York")
ax.legend(frameon=False, ncol=2, markerscale=3)
fig.tight_layout()
fig.savefig(OUT / "mapa-anuncios.png", dpi=180)
plt.close(fig)

# Ausências.
missing = df.isna().sum().sort_values()
missing = missing[missing > 0]
fig, ax = plt.subplots(figsize=(8.8, 4.5))
missing.plot.barh(ax=ax, color=[orange if x > 1000 else blue for x in missing])
ax.set(xlabel="valores ausentes", ylabel="", title="Ausência também é informação sobre a coleta")
for p in ax.patches:
    ax.text(p.get_width() + 150, p.get_y() + p.get_height()/2, f"{int(p.get_width()):,}".replace(",", "."), va="center")
ax.set_xlim(0, 11500)
fig.tight_layout()
fig.savefig(OUT / "valores-ausentes.png", dpi=180)
plt.close(fig)

# Contagens por distrito e tipo.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
df.neighbourhood_group.value_counts().sort_values().plot.barh(ax=axes[0], color=blue)
axes[0].set(title="Anúncios por distrito", xlabel="anúncios", ylabel="")
df.room_type.value_counts().sort_values().plot.barh(ax=axes[1], color=green)
axes[1].set(title="Anúncios por tipo", xlabel="anúncios", ylabel="")
fig.tight_layout()
fig.savefig(OUT / "categorias.png", dpi=180)
plt.close(fig)

# Preço: distribuição bruta e zoom.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
sns.histplot(df.price, bins=60, color=orange, ax=axes[0])
axes[0].set(title="Preço bruto", xlabel="US$ por noite", ylabel="anúncios")
sns.histplot(df.loc[df.price.between(1, 500), "price"], bins=50, color=blue, ax=axes[1])
axes[1].axvline(df.price.median(), color=orange, lw=2, ls="--", label="mediana = US$ 106")
axes[1].set(title="Zoom: US$ 1 a 500", xlabel="US$ por noite", ylabel="")
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "precos-bruto-zoom.png", dpi=180)
plt.close(fig)

# Medianas por distrito e tipo de quarto.
med = df.groupby(["neighbourhood_group", "room_type"], observed=True).price.median().unstack()
order = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
med = med.loc[order]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
med.plot.bar(ax=ax, color=[blue, orange, green])
ax.set(title="Mediana do preço depende de duas categorias", xlabel="", ylabel="US$ por noite")
ax.legend(title="tipo de acomodação", frameon=False)
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(OUT / "mediana-distrito-tipo.png", dpi=180)
plt.close(fig)

print(f"Figuras salvas em {OUT}")
