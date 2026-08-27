from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "exemplos/03-tabelas-tipos/data/AB_NYC_2019.csv"
OUT = ROOT / "slides/assets/aula24-engenharia-atributos-cv"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)
df = df[(df.price > 0) & (df.price <= df.price.quantile(.99))].copy()
sns.set_theme(style="whitegrid", context="talk")
BLUE, ORANGE, GREEN, PURPLE = "#1f6f8b", "#d95f02", "#2a9d8f", "#6a3d9a"
rng = np.random.default_rng(20260827)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Descriptive view
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
sns.boxplot(data=df, x="room_type", y="price", hue="neighbourhood_group", showfliers=False, ax=axes[0])
axes[0].tick_params(axis="x", rotation=18); axes[0].set_title("Preço depende do tipo e da região")
sample = df.sample(7000, random_state=7)
sc = axes[1].scatter(sample.longitude, sample.latitude, c=np.log1p(sample.price), s=7, alpha=.45, cmap="viridis")
axes[1].set(title="Localização contém estrutura", xlabel="longitude", ylabel="latitude")
fig.colorbar(sc, ax=axes[1], label="log(1 + preço)")
save(fig, "base-descritiva.png")

# Log transform
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df.price, bins=60, ax=axes[0], color=ORANGE)
axes[0].set(title="Preço na escala original", xlabel="preço")
sns.histplot(np.log1p(df.price), bins=60, ax=axes[1], color=BLUE)
axes[1].set(title="Após log(1 + preço)", xlabel="log(1 + preço)")
save(fig, "transformacao-log.png")

# Missingness
miss = df[["last_review", "reviews_per_month", "name", "host_name"]].isna().mean().sort_values()
fig, ax = plt.subplots(figsize=(8.5, 4.7)); ax.barh(miss.index, 100*miss.values, color=[GREEN, GREEN, ORANGE, ORANGE]);
ax.set(xlabel="percentual ausente", title="Ausência também pode carregar informação")
for i,v in enumerate(100*miss.values): ax.text(v+.3,i,f"{v:.1f}%",va="center",fontsize=11)
save(fig,"valores-ausentes.png")

# Category means and one-hot motivation
summary = df.groupby(["neighbourhood_group","room_type"], observed=True).price.median().unstack()
fig, ax=plt.subplots(figsize=(10,5)); summary.plot(kind="bar",ax=ax,color=[BLUE,ORANGE,GREEN]);
ax.set(ylabel="mediana do preço",xlabel="região",title="Categorias descrevem grupos com patamares distintos"); ax.tick_params(axis="x",rotation=0)
save(fig,"categorias-preco.png")

# Interaction example
med = df.groupby(["neighbourhood_group","room_type"],observed=True).price.median().reset_index()
fig,ax=plt.subplots(figsize=(9,5)); sns.pointplot(data=med,x="neighbourhood_group",y="price",hue="room_type",markers="o",ax=ax)
ax.set(title="O efeito do tipo de quarto varia por região",xlabel="região",ylabel="mediana do preço"); ax.tick_params(axis="x",rotation=15)
save(fig,"interacao-regiao-quarto.png")


def feature_frames(data):
    base = pd.DataFrame({
        "latitude": data.latitude,
        "longitude": data.longitude,
        "minimum_nights": data.minimum_nights,
        "number_of_reviews": data.number_of_reviews,
        "reviews_per_month": data.reviews_per_month.fillna(0),
        "availability_365": data.availability_365,
    }, index=data.index)
    transformed = base.copy()
    transformed["log_minimum_nights"] = np.log1p(data.minimum_nights)
    transformed["log_number_reviews"] = np.log1p(data.number_of_reviews)
    transformed["reviews_missing"] = data.reviews_per_month.isna().astype(int)
    transformed["name_length"] = data.name.fillna("").str.len()
    categories = pd.concat([
        transformed,
        pd.get_dummies(data[["neighbourhood_group", "room_type"]], drop_first=True, dtype=float)
    ], axis=1)
    enriched = categories.copy()
    lat = data.latitude-data.latitude.mean(); lon=data.longitude-data.longitude.mean()
    enriched["lat2"] = lat**2; enriched["lon2"] = lon**2; enriched["lat_lon"] = lat*lon
    for room in [c for c in categories.columns if c.startswith("room_type_")]:
        enriched[f"{room}:Manhattan"] = categories[room]*(
            data.neighbourhood_group == "Manhattan").astype(float)
    return {"numéricos":base, "+ transformações":transformed,
            "+ categorias":categories, "+ interações":enriched}


def ridge_fit_predict(Xtr, ytr, Xva, lam=1e-3):
    mean=Xtr.mean(0); scale=Xtr.std(0); scale[scale==0]=1
    A=(Xtr-mean)/scale; B=(Xva-mean)/scale; ym=ytr.mean(); yc=ytr-ym
    beta=np.linalg.solve(A.T@A + len(A)*lam*np.eye(A.shape[1]), A.T@yc)
    return ym+B@beta


work = df.sample(12000, random_state=11)
frames = feature_frames(work); y=np.log1p(work.price.to_numpy(float)); n=len(work)
perm=rng.permutation(n); folds=np.array_split(perm,5); scores={k:[] for k in frames}
for k,valid in enumerate(folds):
    train=np.concatenate([f for j,f in enumerate(folds) if j!=k])
    for name,frame in frames.items():
        X=frame.to_numpy(float); pred=ridge_fit_predict(X[train],y[train],X[valid])
        scores[name].append(np.sqrt(np.mean((y[valid]-pred)**2)))

fig,ax=plt.subplots(figsize=(9,5)); positions=np.arange(len(scores)); vals=[scores[k] for k in scores]
ax.boxplot(vals,tick_labels=list(scores),showmeans=True); ax.set(ylabel="RMSE em validação",title="Representações devem ser comparadas fora da amostra"); ax.tick_params(axis="x",rotation=15)
save(fig,"comparacao-features-cv.png")

# Fold diagram as matrix
fold_map=np.full((5,5),0.15); np.fill_diagonal(fold_map,1)
fig,ax=plt.subplots(figsize=(9,5.2)); im=ax.imshow(fold_map,cmap="YlGnBu",vmin=0,vmax=1)
for i in range(5):
    for j in range(5): ax.text(j,i,"Val." if i==j else "Treino",ha="center",va="center",color="white" if i==j else "#18324a",fontsize=12)
ax.set_xticks(range(5),[f"P{i+1}" for i in range(5)]); ax.set_yticks(range(5),[f"Ajuste {i+1}" for i in range(5)])
save(fig,"kfold-esquema.png")

print(pd.DataFrame(scores).agg(["mean","std"]).T.round(4))
