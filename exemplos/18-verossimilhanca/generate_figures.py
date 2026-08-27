from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "exemplos/16-correlacao/data/insurance.csv"
OUT18 = ROOT / "slides/assets/aula18-verossimilhanca"
OUT19 = ROOT / "slides/assets/aula19-verossimilhanca-regressao"
OUT18.mkdir(parents=True, exist_ok=True)
OUT19.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")
COLORS = {"blue": "#1f6f8b", "orange": "#d95f02", "navy": "#17324d", "green": "#2a9d8f"}

df = pd.read_csv(DATA)
z = (df.smoker == "yes").astype(int).to_numpy()
n, s = len(z), z.sum()
p_hat = z.mean()

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def bern_like(p, s, n):
    return np.power(p, s) * np.power(1-p, n-s)

p = np.linspace(.001, .999, 1000)
fig, ax = plt.subplots(figsize=(9, 5))
for xs, label, color in [([1,0], "{1,0}", COLORS["blue"]), ([1,1], "{1,1}", COLORS["orange"]), ([1,0,1], "{1,0,1}", COLORS["green"])]:
    y = bern_like(p, sum(xs), len(xs)); y /= y.max()
    ax.plot(p, y, lw=3, label=label, color=color)
ax.set(xlabel=r"parâmetro candidato $p$", ylabel="verossimilhança relativa", title="Os dados mudam a forma da verossimilhança")
ax.legend(title="amostra")
save(fig, OUT18 / "amostras-mudam-verossimilhanca.png")

fig, ax = plt.subplots(figsize=(9, 5))
ll = s*np.log(p)+(n-s)*np.log(1-p)
ax.plot(p, ll-ll.max(), lw=3, color=COLORS["blue"])
ax.axvline(p_hat, color=COLORS["orange"], ls="--", lw=3, label=fr"$\hat p={p_hat:.3f}$")
ax.set(xlabel=r"$p$", ylabel=r"$\ell(p)-\ell(\hat p)$", title="Log-verossimilhança da proporção de fumantes")
ax.legend()
save(fig, OUT18 / "loglik-fumantes.png")

fig, ax = plt.subplots(figsize=(9, 5))
for nn in [20, 100, 500]:
    ss = round(nn*p_hat); yy = bern_like(p, ss, nn); yy /= yy.max()
    ax.plot(p, yy, lw=3, label=f"n={nn}")
ax.axvline(p_hat, color="black", ls="--")
ax.set(xlabel=r"$p$", ylabel="verossimilhança relativa", title="Mais observações concentram a evidência")
ax.legend()
save(fig, OUT18 / "concentracao-amostra.png")

fig, ax = plt.subplots(figsize=(9, 5))
score = s/p-(n-s)/(1-p)
ax.plot(p, score, lw=3, color=COLORS["blue"])
ax.axhline(0, color="black", lw=1); ax.axvline(p_hat, color=COLORS["orange"], ls="--", lw=3)
ax.set_ylim(-6000,6000); ax.set(xlabel=r"$p$", ylabel=r"escore $U(p)$", title="No máximo, a inclinação da log-verossimilhança é zero")
save(fig, OUT18 / "escore-bernoulli.png")

rng = np.random.default_rng(20260827)
xnorm = rng.normal(12, 2.5, 40)
mu = np.linspace(7,17,500)
for sigma, name in [(2.5,"normal-media")]:
    llmu = -len(xnorm)*np.log(sigma)-np.sum((xnorm[:,None]-mu[None,:])**2,axis=0)/(2*sigma**2)
    fig, ax = plt.subplots(figsize=(9,5)); ax.plot(mu,llmu-llmu.max(),lw=3,color=COLORS["blue"])
    ax.axvline(xnorm.mean(),color=COLORS["orange"],ls="--",lw=3,label=fr"$\hat\mu=\bar x={xnorm.mean():.2f}$")
    ax.set(xlabel=r"média candidata $\mu$",ylabel="log-verossimilhança relativa",title="Na Normal, a média amostral maximiza a verossimilhança"); ax.legend()
    save(fig,OUT18/f"{name}.png")

# Aula 19
x = df.age.to_numpy(float); y = df.charges.to_numpy(float)
xbar,ybar=x.mean(),y.mean(); b1=np.sum((x-xbar)*(y-ybar))/np.sum((x-xbar)**2); b0=ybar-b1*xbar
yhat=b0+b1*x; e=y-yhat; sigma=np.sqrt(np.mean(e**2))

fig, ax = plt.subplots(figsize=(9,5))
ax.scatter(x,y,s=20,alpha=.28,color=COLORS["blue"]); order=np.argsort(x); ax.plot(x[order],yhat[order],lw=3,color=COLORS["orange"])
for i in [25,200,800]: ax.plot([x[i],x[i]],[yhat[i],y[i]],color=COLORS["green"],lw=2)
ax.set(xlabel="idade (anos)",ylabel="despesas (US$)",title="Cada observação é a reta mais um erro vertical")
save(fig,OUT19/"reta-erros.png")

fig, axes = plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
for ax,m,label in zip(axes,[0.35,1,1.65],["inclinação pequena","OLS","inclinação grande"]):
    bb1=m*b1; bb0=ybar-bb1*xbar; pred=bb0+bb1*x; sse=np.sum((y-pred)**2)
    ax.scatter(x,y,s=8,alpha=.18,color=COLORS["blue"]); ax.plot(x[order],pred[order],lw=3,color=COLORS["orange"]); ax.set_title(f"{label}\nSSE={sse/1e9:.1f} bi")
    ax.set_xlabel("idade")
axes[0].set_ylabel("despesas")
save(fig,OUT19/"retas-sse.png")

fig, ax = plt.subplots(figsize=(9,5))
grid=np.linspace(-3.4*sigma,3.4*sigma,600); dens=np.exp(-grid**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
ax.plot(grid,dens,lw=3,color=COLORS["blue"])
for val,c in [(0,COLORS["green"]),(sigma,COLORS["orange"]),(2.3*sigma,"#b2182b")]:
    d=np.exp(-val**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)); ax.vlines(val,0,d,color=c,lw=3); ax.scatter([val],[d],s=90,color=c,zorder=3)
ax.set(xlabel="erro vertical",ylabel="densidade normal",title="Erros menores recebem maior densidade no modelo normal")
save(fig,OUT19/"densidade-erros.png")

scales=np.linspace(.3,1.8,250); sses=[]; lls=[]
for m in scales:
    bb1=m*b1; bb0=ybar-bb1*xbar; ee=y-(bb0+bb1*x); ss=np.sum(ee**2); sses.append(ss); lls.append(-ss/(2*sigma**2))
fig, axes=plt.subplots(1,2,figsize=(12,4.5))
axes[0].plot(scales,sses,lw=3,color=COLORS["orange"]); axes[0].axvline(1,color="black",ls="--"); axes[0].set(xlabel=r"inclinação / $\hat\beta_1$",ylabel="SSE",title="Minimizar SSE")
axes[1].plot(scales,np.array(lls)-max(lls),lw=3,color=COLORS["blue"]); axes[1].axvline(1,color="black",ls="--"); axes[1].set(xlabel=r"inclinação / $\hat\beta_1$",ylabel="log-verossimilhança relativa",title="Maximizar log-verossimilhança")
save(fig,OUT19/"sse-loglik.png")

fig, axes=plt.subplots(1,2,figsize=(12,4.5))
sns.histplot(e,kde=True,ax=axes[0],color=COLORS["blue"]); axes[0].set(title="Resíduos: forte assimetria",xlabel="resíduo")
diagn=pd.DataFrame({"ajustado":yhat,"resíduo":e,"fumante":df.smoker.map({"yes":"Fumante","no":"Não fumante"})})
sns.scatterplot(data=diagn,x="ajustado",y="resíduo",hue="fumante",alpha=.45,ax=axes[1]); axes[1].axhline(0,color="black",ls="--"); axes[1].set_title("A reta omite grupos")
save(fig,OUT19/"diagnostico-normalidade.png")

fig, ax=plt.subplots(figsize=(9,5))
for dist,label,color in [(rng.normal(0,1,5000),"Normal",COLORS["blue"]),(rng.exponential(1,5000)-1,"Assimétrica",COLORS["orange"])]: sns.kdeplot(dist,ax=ax,lw=3,label=label,color=color)
ax.axvline(0,color="black",ls="--"); ax.set(xlabel="erro",ylabel="densidade",title="Média zero não implica distribuição normal"); ax.legend()
save(fig,OUT19/"media-zero-nao-normal.png")

print({"n":n,"smokers":s,"p_hat":p_hat,"b0":b0,"b1":b1,"sigma_mle":sigma})
