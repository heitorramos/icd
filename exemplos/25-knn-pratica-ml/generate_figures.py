from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "exemplos/25-knn-pratica-ml/data/kidney_disease.csv"
OUT = ROOT / "slides/assets/aula25-knn-pratica-ml"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")
BLUE, ORANGE, GREEN, PURPLE = "#1f6f8b", "#d95f02", "#2a9d8f", "#6a3d9a"


def load_clean():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
    df["classification"] = df["classification"].str.replace("\t", "", regex=False).str.strip()
    for col in ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


df = load_clean()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Base description
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
counts = df.classification.value_counts().reindex(["notckd", "ckd"])
axes[0].bar(["sem DRC", "com DRC"], counts, color=[GREEN, ORANGE])
axes[0].set(ylabel="pacientes", title="A resposta é desbalanceada")
missing = df.drop(columns=["id","classification"]).isna().mean().sort_values().tail(10)
axes[1].barh(missing.index, 100*missing.values, color=BLUE)
axes[1].set(xlabel="percentual ausente", title="Ausências fazem parte do problema")
save(fig,"base-descritiva.png")

# 2D intuition using complete hemo/bgr cases
two = df[["hemo","bgr","classification"]].dropna().copy()
query = np.array([12.2, 160.0])
scale = two[["hemo","bgr"]].std().to_numpy(); center=two[["hemo","bgr"]].mean().to_numpy()
dist=np.linalg.norm((two[["hemo","bgr"]].to_numpy()-query)/scale,axis=1)
nearest=np.argsort(dist)[:7]
fig,ax=plt.subplots(figsize=(8.5,5.3)); sns.scatterplot(data=two,x="hemo",y="bgr",hue="classification",palette={"ckd":ORANGE,"notckd":GREEN},alpha=.65,ax=ax)
ax.scatter(*query,s=180,marker="*",color=PURPLE,label="nova paciente")
ax.scatter(two.iloc[nearest].hemo,two.iloc[nearest].bgr,s=110,facecolors="none",edgecolors="black",linewidths=1.5,label="7 vizinhos")
ax.set(xlabel="hemoglobina",ylabel="glicose",title="KNN decide por exemplos próximos"); ax.legend()
save(fig,"knn-intuicao.png")

# Decision regions in standardized two-dimensional data
X2=two[["hemo","bgr"]].to_numpy(); y2=(two.classification=="ckd").astype(int).to_numpy()
mu=X2.mean(0); sd=X2.std(0); Z=(X2-mu)/sd
xmin,xmax=Z[:,0].min()-.4,Z[:,0].max()+.4; ymin,ymax=Z[:,1].min()-.4,Z[:,1].max()+.4
xx,yy=np.meshgrid(np.linspace(xmin,xmax,300),np.linspace(ymin,ymax,300)); grid=np.c_[xx.ravel(),yy.ravel()]
fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharex=True,sharey=True)
for ax,k in zip(axes,[1,7,31]):
    model=KNeighborsClassifier(n_neighbors=k).fit(Z,y2); pred=model.predict(grid).reshape(xx.shape)
    ax.contourf(xx,yy,pred,levels=[-.5,.5,1.5],colors=["#bfe3d7","#f4c7a1"],alpha=.65)
    ax.scatter(Z[:,0],Z[:,1],c=y2,cmap="coolwarm",s=10,alpha=.55); ax.set_title(f"k = {k}"); ax.set_xlabel("hemoglobina padronizada")
axes[0].set_ylabel("glicose padronizada")
save(fig,"fronteiras-k.png")

# Scale illustration
numeric_demo=df[["hemo","bgr","wc","classification"]].dropna().copy(); Xd=numeric_demo[["hemo","bgr","wc"]].to_numpy(float); q=np.nanmedian(Xd,axis=0)
raw=np.sqrt(np.sum((Xd-q)**2,axis=1)); standardized=np.sqrt(np.sum(((Xd-Xd.mean(0))/Xd.std(0)-(q-Xd.mean(0))/Xd.std(0))**2,axis=1))
fig,axes=plt.subplots(1,2,figsize=(11,4.6)); axes[0].bar(["hemo","bgr","wc"],Xd.std(0),color=[GREEN,ORANGE,BLUE]); axes[0].set(yscale="log",ylabel="desvio-padrão (log)",title="Escalas originais são muito diferentes")
axes[1].scatter(raw,standardized,c=(numeric_demo.classification=="ckd"),cmap="coolwarm",s=18,alpha=.6); axes[1].set(xlabel="distância bruta",ylabel="distância padronizada",title="Padronizar muda quem é próximo")
save(fig,"efeito-escala.png")

numeric=["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
categorical=["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
X=df[numeric+categorical]; y=(df.classification=="ckd").astype(int)
Xdev,Xtest,ydev,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=20260827)

preprocess=ColumnTransformer([
    ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
    ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical),
])

# CV curves
folds=StratifiedKFold(5,shuffle=True,random_state=7); rows=[]
for k in range(1,32,2):
    pipe=Pipeline([("prep",preprocess),("knn",KNeighborsClassifier(n_neighbors=k))])
    score=cross_validate(pipe,Xdev,ydev,cv=folds,scoring={"accuracy":"accuracy","recall":"recall","precision":"precision"})
    rows.append({"k":k,"acurácia":score["test_accuracy"].mean(),"recall":score["test_recall"].mean(),"precisão":score["test_precision"].mean(),"sd":score["test_accuracy"].std()})
cv=pd.DataFrame(rows); best_k=int(cv.loc[cv.acurácia.idxmax(),"k"])
fig,ax=plt.subplots(figsize=(9,5)); ax.plot(cv.k,cv.acurácia,marker="o",lw=2.5,label="acurácia",color=BLUE); ax.plot(cv.k,cv.recall,marker="o",lw=2.5,label="recall DRC",color=ORANGE); ax.plot(cv.k,cv.precisão,marker="o",lw=2.5,label="precisão DRC",color=GREEN); ax.axvline(best_k,color="black",ls="--",label=f"melhor acurácia: k={best_k}"); ax.set(xlabel="número de vizinhos k",ylabel="métrica média em validação",ylim=(.75,1.01),title="k muda o compromisso entre métricas"); ax.legend()
save(fig,"validacao-k.png")

# Final fit
final=Pipeline([("prep",preprocess),("knn",KNeighborsClassifier(n_neighbors=best_k))]); final.fit(Xdev,ydev); pred=final.predict(Xtest)
cm=confusion_matrix(ytest,pred); metrics={"acurácia":accuracy_score(ytest,pred),"precisão":precision_score(ytest,pred),"recall":recall_score(ytest,pred),"F1":f1_score(ytest,pred)}
fig,axes=plt.subplots(1,2,figsize=(11,4.7)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["sem DRC","com DRC"],yticklabels=["sem DRC","com DRC"],ax=axes[0]); axes[0].set(xlabel="previsto",ylabel="verdadeiro",title="Teste reservado")
axes[1].bar(metrics.keys(),metrics.values(),color=[BLUE,GREEN,ORANGE,PURPLE]); axes[1].set(ylim=(0,1.05),ylabel="valor",title="Métricas respondem perguntas diferentes"); axes[1].tick_params(axis="x",rotation=20)
save(fig,"teste-metricas.png")

# Error profiles
errors=Xtest.copy(); errors["verdadeiro"]=ytest; errors["previsto"]=pred; errors["erro"]=np.where((ytest==1)&(pred==0),"falso negativo",np.where((ytest==0)&(pred==1),"falso positivo","correto"))
plot=errors[["hemo","bgr","sc","erro"]].melt(id_vars="erro",var_name="atributo",value_name="valor").dropna()
plot["valor_padronizado"]=plot.groupby("atributo")["valor"].transform(lambda s:(s-s.mean())/s.std())
fig,ax=plt.subplots(figsize=(10,5)); sns.boxplot(data=plot,x="atributo",y="valor_padronizado",hue="erro",hue_order=["correto","falso negativo","falso positivo"],palette=[BLUE,ORANGE,GREEN],ax=ax); ax.set(ylabel="valor padronizado no teste",title="Erros aparecem perto de perfis menos típicos"); ax.legend(fontsize=10)
save(fig,"analise-erros.png")

print(cv.round(4).to_string(index=False)); print("best_k",best_k,"metrics",metrics,"cm",cm.tolist())
