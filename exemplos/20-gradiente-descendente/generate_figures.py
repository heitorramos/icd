from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"slides/assets/aula20-gradiente-descendente"
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/"exemplos/16-correlacao/data/insurance.csv")
sns.set_theme(style="whitegrid",context="talk")
blue,orange,green="#1f6f8b","#d95f02","#2a9d8f"
x=df.age.to_numpy(float); y=df.charges.to_numpy(float)
xz=(x-x.mean())/x.std(); yz=(y-y.mean())/y.std()

def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/name,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)

# One-parameter loss and derivative
beta_star=np.sum(xz*yz)/np.sum(xz*xz)
b=np.linspace(-.8,1.4,500)
loss=np.array([np.mean((yz-v*xz)**2) for v in b])
grad=np.array([-2*np.mean(xz*(yz-v*xz)) for v in b])
fig,axes=plt.subplots(1,2,figsize=(12,4.5)); axes[0].plot(b,loss,lw=3,color=blue); axes[0].axvline(beta_star,ls="--",color=orange); axes[0].set(xlabel=r"$\beta$",ylabel="MSE",title="Perda convexa")
axes[1].plot(b,grad,lw=3,color=green); axes[1].axhline(0,color="black"); axes[1].axvline(beta_star,ls="--",color=orange); axes[1].set(xlabel=r"$\beta$",ylabel=r"$dJ/d\beta$",title="O sinal aponta a direção")
save(fig,"perda-derivada.png")

def gd1(lr,steps=25,start=-.7):
    bb=start; hist=[]
    for t in range(steps):
        ll=np.mean((yz-bb*xz)**2); gg=-2*np.mean(xz*(yz-bb*xz)); hist.append((t,bb,ll,gg)); bb-=lr*gg
    return np.array(hist)

fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,lr in zip(axes,[.05,.45,1.05]):
    h=gd1(lr,18); ax.plot(b,loss,color="#bbbbbb",lw=2); ax.plot(h[:,1],h[:,2],"o-",color=orange,lw=2,ms=4); ax.set_title(fr"$\eta={lr}$"); ax.set_xlabel(r"$\beta$")
axes[0].set_ylabel("MSE"); save(fig,"taxas-aprendizado.png")

# 2D contour and path
def mse(a,bb): return np.mean((yz-a-bb*xz)**2)
aa=np.linspace(-1,1,180); bb=np.linspace(-.5,1.1,180); A,B=np.meshgrid(aa,bb); Z=np.vectorize(mse)(A,B)
theta=np.array([.9,-.35]); path=[]; lr=.22
for t in range(24):
    pred=theta[0]+theta[1]*xz; g=np.array([-2*np.mean(yz-pred),-2*np.mean(xz*(yz-pred))]); path.append(theta.copy()); theta-=lr*g
path=np.array(path)
fig,ax=plt.subplots(figsize=(8,5.5)); cs=ax.contour(A,B,Z,levels=18,cmap="Blues"); ax.clabel(cs,inline=True,fontsize=7); ax.plot(path[:,0],path[:,1],"o-",color=orange,lw=2,ms=4); ax.scatter([0],[beta_star],s=100,color=green,zorder=4); ax.set(xlabel=r"intercepto padronizado $\alpha$",ylabel=r"inclinação $\beta$",title="Gradiente descendente percorre a superfície de perda"); save(fig,"contorno-caminho.png")

# convergence and OLS
h=gd1(.25,50)
fig,axes=plt.subplots(1,2,figsize=(12,4.5)); axes[0].plot(h[:,0],h[:,1],lw=3,color=blue); axes[0].axhline(beta_star,ls="--",color=orange,label="solução OLS"); axes[0].set(xlabel="iteração",ylabel=r"$\beta_t$",title="O parâmetro converge"); axes[0].legend()
axes[1].semilogy(h[:,0],h[:,2]-loss.min()+1e-12,lw=3,color=green); axes[1].set(xlabel="iteração",ylabel="excesso de perda (log)",title="A perda estabiliza"); save(fig,"convergencia-ols.png")

# SGD vs batch
rng=np.random.default_rng(20260827)
def train(batch,epochs=35,lr=.04):
    th=np.zeros(2); out=[]
    for epoch in range(epochs):
        ids=rng.permutation(len(xz))
        for k in range(0,len(ids),batch):
            ii=ids[k:k+batch]; pred=th[0]+th[1]*xz[ii]; g=np.array([-2*np.mean(yz[ii]-pred),-2*np.mean(xz[ii]*(yz[ii]-pred))]); th-=lr*g
        out.append((epoch,mse(*th),*th))
    return np.array(out)
fig,ax=plt.subplots(figsize=(9,5))
for batch,label,c in [(len(xz),"batch completo",blue),(64,"mini-batch 64",green),(1,"SGD",orange)]:
    hh=train(batch); ax.plot(hh[:,0],hh[:,1],lw=2.5,label=label,color=c)
ax.set(xlabel="época",ylabel="MSE",title="Mini-batches trocam suavidade por atualizações baratas"); ax.legend(); save(fig,"batch-sgd.png")

# scaling effect synthetic unscaled contour paths
age=x; charges=y
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
axes[0].scatter(age,charges,s=10,alpha=.25,color=blue); axes[0].set(xlabel="idade",ylabel="despesas",title="Escalas originais")
axes[1].scatter(xz,yz,s=10,alpha=.25,color=green); axes[1].set(xlabel="idade padronizada",ylabel="despesa padronizada",title="Escalas comparáveis")
save(fig,"padronizacao.png")

# neural loss surface conceptual
w=np.linspace(-2.5,2.5,250); L=(w**2-1.2)**2+.15*w
fig,ax=plt.subplots(figsize=(9,5)); ax.plot(w,L,lw=3,color=blue); ax.scatter([w[np.argmin(L)]],[L.min()],s=110,color=green,label="mínimo global"); loc=np.where((np.r_[False,np.diff(np.sign(np.diff(L)))>0,False]))[0]; ax.scatter(w[loc],L[loc],s=80,color=orange,label="mínimos locais"); ax.set(xlabel="um peso da rede",ylabel="perda",title="Redes neurais produzem superfícies não convexas"); ax.legend(); save(fig,"nao-convexa-rede.png")

print({"beta_ols_padronizado":beta_star})
