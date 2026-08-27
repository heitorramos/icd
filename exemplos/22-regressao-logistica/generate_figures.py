from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"slides/assets/aula22-regressao-logistica"; OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/"exemplos/16-correlacao/data/insurance.csv"); sns.set_theme(style="whitegrid",context="talk"); blue,orange,green="#1f6f8b","#d95f02","#2a9d8f"
def save(fig,n): fig.tight_layout(); fig.savefig(OUT/n,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)
eta=np.linspace(-6,6,600)
funcs={"linear":.5+.15*eta,"degrau":(eta>=0).astype(float),"arctan":.5+np.arctan(eta)/np.pi,"logística":1/(1+np.exp(-eta))}
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for ax,(name,val) in zip(axes.flat,funcs.items()): ax.plot(eta,val,lw=3,color=blue if name!="logística" else orange); ax.axhline(0,color="black",lw=.7); ax.axhline(1,color="black",lw=.7); ax.set_title(name); ax.set_ylim(-.25,1.25)
save(fig,"funcoes-candidatas.png")
fig,ax=plt.subplots(figsize=(9,5));
for b,l,c in [(.5,"inclinação 0,5",green),(1,"inclinação 1",blue),(2,"inclinação 2",orange)]: ax.plot(eta,1/(1+np.exp(-b*eta)),lw=3,label=l,color=c)
ax.set(xlabel=r"preditor linear $\eta$",ylabel="probabilidade",title="Coeficientes controlam a transição da sigmoide"); ax.legend(); save(fig,"sigmoides.png")
y=(df.smoker=="yes").astype(int).to_numpy(); cols=["age","bmi","charges"]; Xraw=df[cols].to_numpy(float); mean=Xraw.mean(0); sd=Xraw.std(0); Z=(Xraw-mean)/sd; X=np.column_stack([np.ones(len(df)),Z])
def sig(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
b=np.zeros(X.shape[1]); hist=[]
for t in range(2500):
 p=sig(X@b); loss=-np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)); g=X.T@(p-y)/len(y); hist.append(loss); b-=.15*g
p=sig(X@b); pred=(p>=.5).astype(int)
fig,axes=plt.subplots(1,2,figsize=(12,4.5)); jitter=np.random.default_rng(1).normal(0,.025,len(y)); axes[0].scatter(df.charges,y+jitter,c=y,cmap="coolwarm",s=10,alpha=.35); axes[0].set(xlabel="despesas (US$)",ylabel="fumante",title="Resposta binária com sobreposição"); prob_df=pd.DataFrame({"p":p,"classe":y}); sns.histplot(data=prob_df,x="p",hue="classe",bins=25,element="step",stat="density",common_norm=False,ax=axes[1]); axes[1].set(xlabel="probabilidade prevista",title="Distribuições das probabilidades"); save(fig,"base-probabilidades.png")
fig,ax=plt.subplots(figsize=(9,5)); ax.semilogy(hist,lw=3,color=blue); ax.set(xlabel="iteração",ylabel="entropia cruzada",title="Gradiente descendente reduz a log-loss"); save(fig,"convergencia-logistica.png")
pp=np.linspace(.001,.999,500); fig,axes=plt.subplots(1,2,figsize=(12,4.5)); axes[0].plot(pp,-np.log(pp),lw=3,color=green); axes[0].set(xlabel="probabilidade prevista para classe 1",ylabel="perda",title="Quando y=1"); axes[1].plot(pp,-np.log(1-pp),lw=3,color=orange); axes[1].set(xlabel="probabilidade prevista para classe 1",ylabel="perda",title="Quando y=0"); save(fig,"perda-logistica.png")
# calibration bins
bins=pd.qcut(p,10,duplicates="drop"); cal=pd.DataFrame({"p":p,"y":y,"bin":bins}).groupby("bin",observed=True).agg(prev=("y","mean"),pred=("p","mean"),n=("y","size"))
fig,ax=plt.subplots(figsize=(7,6)); ax.plot([0,1],[0,1],"--",color="black"); ax.scatter(cal.pred,cal.prev,s=cal.n*1.4,color=blue,alpha=.7); ax.set(xlim=(0,1),ylim=(0,1),xlabel="probabilidade média prevista",ylabel="frequência observada",title="Calibração compara probabilidades com frequências"); save(fig,"calibracao.png")
# confusion at thresholds
thresholds=np.linspace(.05,.95,19); sens=[]; spec=[]
for th in thresholds:
 q=p>=th; sens.append(((q==1)&(y==1)).sum()/(y==1).sum()); spec.append(((q==0)&(y==0)).sum()/(y==0).sum())
fig,ax=plt.subplots(figsize=(9,5)); ax.plot(thresholds,sens,lw=3,label="sensibilidade",color=orange); ax.plot(thresholds,spec,lw=3,label="especificidade",color=green); ax.axvline(.5,ls="--",color="black"); ax.set(xlabel="limiar",ylabel="proporção",title="O limiar é uma decisão, não parte do modelo probabilístico"); ax.legend(); save(fig,"limiares.png")
print({"beta":b,"accuracy":(pred==y).mean(),"prevalence":y.mean()})
