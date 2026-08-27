from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides/assets/aula26-pca-svm"
OUT.mkdir(parents=True, exist_ok=True)
data = np.load(ROOT / "exemplos/26-pca-svm/data/fashion_mnist_sample.npz")
images, labels = data["images"], data["labels"]
names = np.array(["Camiseta", "Calça", "Suéter", "Vestido", "Casaco",
                  "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"])
sns.set_theme(style="whitegrid", context="talk")
BLUE, ORANGE, GREEN, PURPLE = "#1f6f8b", "#d95f02", "#2a9d8f", "#6a3d9a"
rng = np.random.default_rng(20260827)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Cat: low-rank approximation after resizing to keep the notebook lightweight.
cat = Image.open(OUT / "cat-original.jpeg").convert("L")
cat.thumbnail((420, 280))
A = np.asarray(cat, dtype=float) / 255
U, s, Vt = np.linalg.svd(A, full_matrices=False)
ranks = [5, 20, 60, 120]
fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
flat = axes.flat
flat[0].imshow(A, cmap="gray", vmin=0, vmax=1); flat[0].set_title("Original")
for ax, k in zip(list(flat)[1:5], ranks):
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    ax.imshow(Ak, cmap="gray", vmin=0, vmax=1); ax.set_title(f"posto {k}")
for ax in axes.flat: ax.axis("off")
save(fig, "gato-postos.png")

energy = np.cumsum(s**2)/np.sum(s**2)
storage = np.array([(k*(A.shape[0]+A.shape[1]+1))/(A.shape[0]*A.shape[1]) for k in range(1,len(s)+1)])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].semilogy(np.arange(1,len(s)+1), s, color=BLUE, lw=2.5)
axes[0].set(xlabel="componente", ylabel="valor singular", title="O espectro decai rapidamente")
axes[1].plot(storage, energy, color=ORANGE, lw=3)
axes[1].axhline(.9, color="black", ls="--"); axes[1].axvline(1, color="grey", ls=":")
axes[1].set(xlabel="fração de armazenamento", ylabel="energia acumulada", title="Qualidade versus armazenamento")
save(fig, "gato-espectro-compressao.png")

# Fashion-MNIST examples
fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
for label, ax in enumerate(axes.flat):
    i = np.where(labels == label)[0][0]
    ax.imshow(images[i], cmap="gray"); ax.set_title(names[label]); ax.axis("off")
save(fig, "fashion-classes.png")

X = images.reshape(len(images), -1).astype(float)/255
perm = rng.permutation(len(X)); cut = 4800
train, test = perm[:cut], perm[cut:]
Xtr, Xte, ytr, yte = X[train], X[test], labels[train], labels[test]

pca_full = PCA(n_components=150, svd_solver="randomized", random_state=7).fit(Xtr)
cum = np.cumsum(pca_full.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(8.5, 4.8)); ax.plot(np.arange(1,len(cum)+1),cum,lw=3,color=BLUE)
for level,color in [(.80,GREEN),(.90,ORANGE),(.95,PURPLE)]:
    k=int(np.searchsorted(cum,level)+1); ax.axhline(level,color=color,ls="--"); ax.axvline(k,color=color,ls=":"); ax.text(k+2,level-.035,f"{int(level*100)}%: {k} comps.",color=color,fontsize=11)
ax.set(xlabel="número de componentes",ylabel="variância explicada acumulada",title="PCA concentra variação em menos dimensões",ylim=(0,1.01))
save(fig,"pca-variancia.png")

pca2=PCA(n_components=2,random_state=7); Z=pca2.fit_transform(Xtr)
sel=rng.choice(len(Z),2200,replace=False)
fig,ax=plt.subplots(figsize=(9,5.5)); sc=ax.scatter(Z[sel,0],Z[sel,1],c=ytr[sel],s=9,alpha=.55,cmap="tab10")
ax.set(xlabel="PC1",ylabel="PC2",title="Duas componentes revelam estrutura, não separam tudo")
handles=[plt.Line2D([],[],marker="o",ls="",color=plt.cm.tab10(i/9),label=names[i]) for i in range(10)]
ax.legend(handles=handles,ncol=2,fontsize=8,loc="best")
save(fig,"pca-projecao-2d.png")

# Reconstructions
fig, axes=plt.subplots(4,6,figsize=(11,7))
examples=[np.where(yte==i)[0][0] for i in [0,2,6,9]]
components=[None,5,20,50,100,150]
for r,idx in enumerate(examples):
    axes[r,0].imshow(Xte[idx].reshape(28,28),cmap="gray",vmin=0,vmax=1); axes[r,0].set_ylabel(names[yte[idx]],fontsize=10)
    for c,k in enumerate(components[1:],start=1):
        rec=pca_full.mean_ + (pca_full.transform(Xte[idx:idx+1])[:,:k] @ pca_full.components_[:k])[0]
        axes[r,c].imshow(rec.reshape(28,28),cmap="gray",vmin=0,vmax=1)
for c,k in enumerate(components): axes[0,c].set_title("Original" if k is None else f"{k} PCs",fontsize=11)
for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
save(fig,"fashion-reconstrucoes.png")

# SVM margin on a synthetic 2D projection of two classes
mask=np.isin(ytr,[0,9]); X2=Z[mask]; y2=np.where(ytr[mask]==9,1,-1)
linear=SVC(kernel="linear",C=1).fit(X2,y2)
xg=np.linspace(X2[:,0].min(),X2[:,0].max(),250); yg=np.linspace(X2[:,1].min(),X2[:,1].max(),250); XX,YY=np.meshgrid(xg,yg)
decision=linear.decision_function(np.c_[XX.ravel(),YY.ravel()]).reshape(XX.shape)
fig,ax=plt.subplots(figsize=(8.5,5.3)); ax.scatter(X2[:,0],X2[:,1],c=y2,cmap="coolwarm",s=10,alpha=.4)
ax.contour(XX,YY,decision,levels=[-1,0,1],colors=["grey","black","grey"],linestyles=["--","-","--"])
sv=linear.support_vectors_; ax.scatter(sv[:,0],sv[:,1],s=45,facecolors="none",edgecolors="black",label="vetores de suporte")
ax.set(xlabel="PC1",ylabel="PC2",title="A margem depende dos pontos mais próximos"); ax.legend()
save(fig,"svm-margem.png")

# Pipeline comparisons and confusion matrix
results=[]
for ncomp in [20,50,100,150]:
    model=Pipeline([("pca",PCA(n_components=ncomp,svd_solver="randomized",random_state=7)),("svm",SVC(C=10,kernel="rbf",gamma="scale"))])
    t0=perf_counter(); model.fit(Xtr,ytr); fit_time=perf_counter()-t0
    pred=model.predict(Xte); results.append((ncomp,accuracy_score(yte,pred),fit_time,model,pred))
res=pd.DataFrame([(k,a,t) for k,a,t,_,_ in results],columns=["componentes","acurácia","tempo"])
fig,axes=plt.subplots(1,2,figsize=(11,4.5)); axes[0].plot(res.componentes,res.acurácia,marker="o",lw=3,color=BLUE); axes[0].set(xlabel="componentes PCA",ylabel="acurácia",title="Compressão pode preservar desempenho")
axes[1].plot(res.componentes,res.tempo,marker="o",lw=3,color=ORANGE); axes[1].set(xlabel="componentes PCA",ylabel="tempo de ajuste (s)",title="Mais dimensões custam mais")
save(fig,"pca-svm-desempenho.png")

best=max(results,key=lambda x:x[1]); cm=confusion_matrix(yte,best[4],normalize="true")
fig,ax=plt.subplots(figsize=(8,6.5)); sns.heatmap(cm,cmap="Blues",vmin=0,vmax=1,xticklabels=names,yticklabels=names,ax=ax)
ax.set(xlabel="previsto",ylabel="verdadeiro",title=f"SVM após PCA ({best[0]} componentes)"); ax.tick_params(axis="x",rotation=35); ax.tick_params(axis="y",rotation=0)
save(fig,"matriz-confusao.png")

# CV grid for applied hyperparameter selection
subset=rng.choice(len(Xtr),3000,replace=False); skf=StratifiedKFold(3,shuffle=True,random_state=7)
components_grid=[30,60,100]; C_grid=[1,10,30]; grid=np.zeros((len(components_grid),len(C_grid)))
for i,k in enumerate(components_grid):
    for j,C in enumerate(C_grid):
        fold=[]
        for a,b in skf.split(Xtr[subset],ytr[subset]):
            pipe=Pipeline([("pca",PCA(n_components=k,svd_solver="randomized",random_state=7)),("svm",SVC(C=C,kernel="rbf",gamma="scale"))])
            pipe.fit(Xtr[subset][a],ytr[subset][a]); fold.append(pipe.score(Xtr[subset][b],ytr[subset][b]))
        grid[i,j]=np.mean(fold)
fig,ax=plt.subplots(figsize=(7.5,5.2)); sns.heatmap(grid,annot=True,fmt=".3f",cmap="YlGnBu",xticklabels=C_grid,yticklabels=components_grid,ax=ax)
ax.set(xlabel="C do SVM",ylabel="componentes PCA",title="A pipeline inteira é escolhida por validação")
save(fig,"cv-pca-svm.png")

print(res.round(4).to_string(index=False))
print("best",best[0],best[1])
