from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "exemplos/extra-dados-ausentes/data/kidney_disease.csv"
OUT = ROOT / "slides/assets/extra-dados-ausentes"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")
BLUE, ORANGE, GREEN, PURPLE, RED = "#1f6f8b", "#d95f02", "#2a9d8f", "#6a3d9a", "#b23a48"


def load_clean():
    df = pd.read_csv(DATA)
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
    df["classification"] = df["classification"].str.replace("\t", "", regex=False).str.strip()
    numeric = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, numeric


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


df, numeric = load_clean()
target = (df.classification == "ckd").astype(int)

# Missingness overview
miss = df.drop(columns=["id", "classification"]).isna().mean().sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].barh(miss.head(14).index[::-1], 100 * miss.head(14).values[::-1], color=BLUE)
axes[0].set(xlabel="percentual ausente", title="Ausência varia entre atributos")
rowmiss = df.drop(columns=["id", "classification"]).isna().sum(axis=1)
axes[1].hist(rowmiss, bins=np.arange(rowmiss.max() + 2) - .5, color=ORANGE, edgecolor="white")
axes[1].set(xlabel="atributos ausentes por paciente", ylabel="pacientes", title="Ausências também se acumulam por linha")
save(fig, "panorama-ausencias.png")

# Missingness matrix, ordered by class and number missing
cols = miss.head(12).index.tolist()
order = pd.DataFrame({"class": target, "n": df[cols].isna().sum(axis=1)}).sort_values(["class", "n"]).index
matrix = df.loc[order, cols].isna().astype(int)
fig, ax = plt.subplots(figsize=(10.5, 5.2))
sns.heatmap(matrix.T, cmap=["#edf4f7", RED], cbar=False, xticklabels=False, ax=ax)
ax.set(xlabel="pacientes ordenados por classe e número de ausências", ylabel="atributo", title="Ausências formam padrões, não apenas células isoladas")
save(fig, "mapa-ausencias.png")

# Missingness by target
by = pd.DataFrame({
    "com DRC": df.loc[target == 1, numeric].isna().mean(),
    "sem DRC": df.loc[target == 0, numeric].isna().mean(),
}).sort_values("com DRC", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 5.2))
by.iloc[::-1].mul(100).plot.barh(ax=ax, color=[ORANGE, GREEN])
ax.set(xlabel="percentual ausente", title="A ausência está associada à resposta observada?")
save(fig, "ausencia-por-classe.png")

# Complete-case selection
complete = df[numeric].notna().all(axis=1)
summary = pd.DataFrame({
    "amostra completa": [target.mean(), df.age.mean(), df.hemo.mean()],
    "casos completos": [target[complete].mean(), df.loc[complete, "age"].mean(), df.loc[complete, "hemo"].mean()],
}, index=["proporção DRC", "idade média / 100", "hemoglobina média / 20"])
summary.loc["idade média / 100"] /= 100
summary.loc["hemoglobina média / 20"] /= 20
fig, ax = plt.subplots(figsize=(9.5, 5))
summary.plot.bar(ax=ax, color=[BLUE, ORANGE])
ax.set(ylabel="valor reescalado", title=f"Casos completos mudam a composição (n={complete.sum()} de {len(df)})")
ax.tick_params(axis="x", rotation=0)
save(fig, "casos-completos.png")

# Simulation: mechanisms and bias
rng = np.random.default_rng(20260827)
n, reps = 1500, 500
rows = []
for mechanism in ["MCAR", "MAR", "MNAR"]:
    for _ in range(reps):
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        if mechanism == "MCAR":
            prob = np.repeat(.35, n)
        elif mechanism == "MAR":
            prob = 1 / (1 + np.exp(-(-.7 + 1.2 * z)))
        else:
            prob = 1 / (1 + np.exp(-(-.7 + 1.2 * x)))
        observed = rng.random(n) > prob
        rows.append((mechanism, x[observed].mean() - x.mean()))
sim = pd.DataFrame(rows, columns=["mecanismo", "erro"])
fig, ax = plt.subplots(figsize=(9.5, 5))
sns.violinplot(data=sim, x="mecanismo", y="erro", hue="mecanismo", palette=[GREEN, BLUE, ORANGE], legend=False, inner="quartile", ax=ax)
ax.axhline(0, color="black", lw=1)
ax.set(ylabel="média observada − média completa", title="Ignorar ausências pode produzir viés sistemático")
save(fig, "simulacao-mecanismos.png")

# Imputation comparison in prediction
categorical = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
work = df[numeric + categorical].copy()
for col in categorical:
    work[col] = work[col].astype(object).where(work[col].notna(), np.nan)
cv = StratifiedKFold(5, shuffle=True, random_state=7)
rows = []
for strategy in ["mean", "median"]:
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy=strategy)), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=3000))])
    score = cross_validate(pipe, work, target, cv=cv, scoring=["roc_auc", "recall", "accuracy"])
    for metric in ["roc_auc", "recall", "accuracy"]:
        rows.append((strategy, metric, score[f"test_{metric}"].mean(), score[f"test_{metric}"].std()))
res = pd.DataFrame(rows, columns=["estratégia", "métrica", "média", "sd"])
fig, ax = plt.subplots(figsize=(9.5, 5))
sns.barplot(data=res, x="métrica", y="média", hue="estratégia", palette=[BLUE, ORANGE], ax=ax)
ax.set(ylim=(.85, 1.01), ylabel="média em validação", title="Desempenho parecido não torna a hipótese equivalente")
save(fig, "comparacao-imputacao.png")

# Sensitivity delta adjustment
hemo = df.hemo.copy()
base = hemo.fillna(hemo.median())
deltas = np.linspace(-3, 3, 61)
means = [(base.where(hemo.notna(), base + delta)).mean() for delta in deltas]
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(deltas, means, color=PURPLE, lw=3)
ax.axvline(0, color="black", ls="--")
ax.set(xlabel=r"ajuste $\delta$ nos valores imputados", ylabel="hemoglobina média", title="Análise de sensibilidade: e se os ausentes forem sistematicamente diferentes?")
save(fig, "sensibilidade-delta.png")

print("n_complete", int(complete.sum()))
print("missing_top", miss.head(8).round(3).to_dict())
print(res.round(4).to_string(index=False))
