from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides/assets/aula23-regularizacao"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "exemplos/16-correlacao/data/insurance.csv")
sns.set_theme(style="whitegrid", context="talk")
BLUE, ORANGE, GREEN, PURPLE = "#1f6f8b", "#d95f02", "#2a9d8f", "#6a3d9a"
rng = np.random.default_rng(20260827)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def standardize_train_test(Xtr, Xte):
    mean, scale = Xtr.mean(0), Xtr.std(0)
    scale[scale == 0] = 1
    return (Xtr-mean)/scale, (Xte-mean)/scale


def ridge(X, y, alpha):
    return np.linalg.solve(X.T@X + alpha*np.eye(X.shape[1]), X.T@y)


age = df.age.to_numpy(float)
y = df.charges.to_numpy(float)
smoker = (df.smoker == "yes").to_numpy(float)
bmi = df.bmi.to_numpy(float)
idx = rng.permutation(len(df)); cut = int(.75*len(df)); tr, te = idx[:cut], idx[cut:]

# Data overview
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=.45, ax=axes[0])
axes[0].set_title("Despesas dependem de vários fatores")
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", alpha=.45, ax=axes[1], legend=False)
axes[1].set_title("Há grupos e relações não lineares")
save(fig, "base-descritiva.png")

# Polynomial overfit illustration
x = np.linspace(-1, 1, 24); truth = 5 + 3*x + 2*x**2
y_small = truth + rng.normal(0, 1.2, len(x)); grid = np.linspace(-1.08, 1.08, 400)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, degree, color in zip(axes, [1, 2, 15], [ORANGE, GREEN, PURPLE]):
    coef = np.polyfit(x, y_small, degree); pred = np.polyval(coef, grid)
    ax.scatter(x, y_small, s=25, alpha=.7, color=BLUE); ax.plot(grid, pred, lw=3, color=color)
    ax.plot(grid, 5+3*grid+2*grid**2, "--", lw=2, color="black", alpha=.6)
    ax.set(title=f"grau {degree}", xlabel="x", ylim=(0, 14))
axes[0].set_ylabel("y")
save(fig, "complexidade-polinomial.png")

# Bias variance simulated
grid = np.linspace(-1, 1, 160); truth_grid = 5+3*grid+2*grid**2
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, degree, color in zip(axes, [1, 2, 12], [ORANGE, GREEN, PURPLE]):
    predictions=[]
    for _ in range(70):
        yy=truth+rng.normal(0,1.2,len(x)); predictions.append(np.polyval(np.polyfit(x,yy,degree),grid))
    predictions=np.array(predictions)
    for p in predictions[:18]: ax.plot(grid,p,color=color,alpha=.13,lw=1)
    ax.plot(grid,predictions.mean(0),color=color,lw=3,label="média dos ajustes")
    ax.plot(grid,truth_grid,"--",color="black",lw=2,label="função real")
    ax.set(title=f"grau {degree}",xlabel="x",ylim=(0,14))
axes[0].set_ylabel("y"); axes[-1].legend(fontsize=10)
save(fig,"vies-variancia.png")

# Rich feature matrix for insurance
raw = np.column_stack([age, bmi, smoker, age*bmi, age*smoker, bmi*smoker,
                       age**2, bmi**2, age**3, bmi**3])
names = ["age", "bmi", "smoker", "age:bmi", "age:smoker", "bmi:smoker",
         "age²", "bmi²", "age³", "bmi³"]
Xtr, Xte = standardize_train_test(raw[tr], raw[te]); ymean=y[tr].mean(); yc=y[tr]-ymean
alphas=np.logspace(-3,5,120); coefs=np.array([ridge(Xtr,yc,a) for a in alphas])
rmse_tr=np.array([np.sqrt(np.mean((yc-Xtr@b)**2)) for b in coefs])
rmse_te=np.array([np.sqrt(np.mean((y[te]-(ymean+Xte@b))**2)) for b in coefs])

fig, ax=plt.subplots(figsize=(9,5.2))
for j,n in enumerate(names): ax.semilogx(alphas,coefs[:,j],lw=2,label=n)
ax.axhline(0,color="black",lw=1); ax.set(xlabel=r"penalização $\lambda$",ylabel="coeficiente",title="Ridge contrai coeficientes continuamente")
ax.legend(ncol=2,fontsize=9)
save(fig,"caminho-ridge.png")

fig,ax=plt.subplots(figsize=(9,5.2)); ax.semilogx(alphas,rmse_tr,lw=3,label="treino",color=BLUE); ax.semilogx(alphas,rmse_te,lw=3,label="validação",color=ORANGE)
best=np.argmin(rmse_te); ax.axvline(alphas[best],ls="--",color="black",label=fr"melhor $\lambda$={alphas[best]:.2g}")
ax.set(xlabel=r"penalização $\lambda$",ylabel="RMSE",title="Validação escolhe a complexidade"); ax.legend()
save(fig,"curva-validacao.png")

# Geometry
t=np.linspace(0,2*np.pi,500); fig,axes=plt.subplots(1,2,figsize=(11,5))
u=np.linspace(-3,3,350); U,V=np.meshgrid(u,u); loss=(U-2.0)**2+2*(V-1.3)**2+.5*(U-2)*(V-1.3)
for ax,kind in zip(axes,["L1","L2"]):
    ax.contour(U,V,loss,levels=12,cmap="Blues")
    if kind=="L1":
        d=np.linspace(-1.8,1.8,200); ax.plot(d,1.8-np.abs(d),color=ORANGE,lw=3); ax.plot(d,-1.8+np.abs(d),color=ORANGE,lw=3)
    else: ax.plot(1.8*np.cos(t),1.8*np.sin(t),color=GREEN,lw=3)
    ax.axhline(0,color="grey",lw=.8); ax.axvline(0,color="grey",lw=.8); ax.set(aspect="equal",xlim=(-3,3),ylim=(-3,3),title=f"restrição {kind}",xlabel=r"$\beta_1$",ylabel=r"$\beta_2$")
save(fig,"geometria-l1-l2.png")

# Stability under resampling: OLS vs ridge
ols=[]; rid=[]
for _ in range(500):
    ii=rng.integers(0,len(tr),len(tr)); Xi=Xtr[ii]; yi=yc[ii]
    ols.append(np.linalg.lstsq(Xi,yi,rcond=None)[0]); rid.append(ridge(Xi,yi,alphas[best]))
ols=np.array(ols); rid=np.array(rid)
fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
sel=[0,1,3,6]
axes[0].boxplot([ols[:,j] for j in sel],tick_labels=[names[j] for j in sel]); axes[0].set_title("Sem regularização")
axes[1].boxplot([rid[:,j] for j in sel],tick_labels=[names[j] for j in sel]); axes[1].set_title("Ridge")
for ax in axes: ax.tick_params(axis="x",rotation=20); ax.axhline(0,color="black",lw=.8)
axes[0].set_ylabel("coeficiente em reamostras")
save(fig,"estabilidade-coeficientes.png")

print({"lambda_best":float(alphas[best]),"rmse_train":float(rmse_tr[best]),"rmse_valid":float(rmse_te[best])})
