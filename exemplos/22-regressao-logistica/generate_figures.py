from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"slides/assets/aula22-regressao-logistica"; OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/"exemplos/16-correlacao/data/insurance.csv"); sns.set_theme(style="whitegrid",context="talk"); blue,orange,green="#1f6f8b","#d95f02","#2a9d8f"
def save(fig,n): fig.tight_layout(); fig.savefig(OUT/n,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)
eta=np.linspace(-6,6,600)
funcs={"linear":.5+.15*eta,"degrau":(eta>=0).astype(float),"arctan":.5+np.arctan(eta)/np.pi,"logística":1/(1+np.exp(-eta))}

# Identity and inverse-logit links used in the GLM introduction.
sigmoid=1/(1+np.exp(-eta))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
axes[0].plot(eta,eta,lw=3,color=blue)
axes[0].set(xlabel=r"preditor linear $\eta$",ylabel=r"média $\mu$",title="Normal: ligação identidade")
axes[1].plot(eta,sigmoid,lw=3,color=orange)
axes[1].axhline(.5,color="black",ls="--")
axes[1].set(xlabel=r"preditor linear $\eta$",ylabel=r"probabilidade $\mu$",title="Bernoulli: inversa da ligação logit")
save(fig,"links-glm.png")

fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for ax,(name,val) in zip(axes.flat,funcs.items()): ax.plot(eta,val,lw=3,color=blue if name!="logística" else orange); ax.axhline(0,color="black",lw=.7); ax.axhline(1,color="black",lw=.7); ax.set_title(name); ax.set_ylim(-.25,1.25)
save(fig,"funcoes-candidatas.png")
fig,ax=plt.subplots(figsize=(9,5));
for b,l,c in [(.5,"inclinação 0,5",green),(1,"inclinação 1",blue),(2,"inclinação 2",orange)]: ax.plot(eta,1/(1+np.exp(-b*eta)),lw=3,label=l,color=c)
ax.set(xlabel=r"preditor linear $\eta$",ylabel="probabilidade",title="Coeficientes controlam a transição da sigmoide"); ax.legend(); save(fig,"sigmoides.png")
y=(df.smoker=="yes").astype(int).to_numpy(); cols=["age","bmi","charges"]; Xraw=df[cols].to_numpy(float); mean=Xraw.mean(0); sd=Xraw.std(0); Z=(Xraw-mean)/sd; X=np.column_stack([np.ones(len(df)),Z])
def sig(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

# A nonlinear boundary obtained by expanding the input features while keeping
# the logistic model linear in its coefficients.
rng_nl=np.random.default_rng(20261112)
X_nl=rng_nl.uniform(-2.4,2.4,size=(360,2))
score_nl=1.0-.65*X_nl[:,0]**2-.85*X_nl[:,1]+.25*X_nl[:,0]
y_nl=rng_nl.binomial(1,sig(2.7*score_nl))

def nonlinear_design(points,quadratic=False):
    x1,x2=points[:,0],points[:,1]
    columns=[np.ones(len(points)),x1,x2]
    if quadratic: columns.extend([x1**2,x1*x2,x2**2])
    return np.column_stack(columns)

def fit_logistic_design(design,response,learning_rate,steps=7000):
    coefficients=np.zeros(design.shape[1])
    for _ in range(steps):
        probability=sig(design@coefficients)
        coefficients-=learning_rate*(design.T@(probability-response)/len(response))
    return coefficients

beta_nl_linear=fit_logistic_design(nonlinear_design(X_nl),y_nl,.12)
beta_nl_quadratic=fit_logistic_design(nonlinear_design(X_nl,True),y_nl,.025)
grid_axis=np.linspace(-2.5,2.5,240)
grid_x1,grid_x2=np.meshgrid(grid_axis,grid_axis)
grid_points=np.column_stack([grid_x1.ravel(),grid_x2.ravel()])
probability_grids=[
    sig(nonlinear_design(grid_points)@beta_nl_linear).reshape(grid_x1.shape),
    sig(nonlinear_design(grid_points,True)@beta_nl_quadratic).reshape(grid_x1.shape),
]
predictions=[
    sig(nonlinear_design(X_nl)@beta_nl_linear)>=.5,
    sig(nonlinear_design(X_nl,True)@beta_nl_quadratic)>=.5,
]
fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharex=True,sharey=True)
for ax,probability_grid,prediction,title in zip(
    axes,probability_grids,predictions,
    [r"Apenas $x_1$ e $x_2$: fronteira reta",r"Com termos quadráticos: fronteira curva"]
):
    ax.contourf(grid_x1,grid_x2,probability_grid,levels=[0,.5,1],colors=["#f7c9b2","#b9dce5"],alpha=.45)
    ax.contour(grid_x1,grid_x2,probability_grid,levels=[.5],colors=["#25364a"],linewidths=3)
    ax.scatter(X_nl[y_nl==0,0],X_nl[y_nl==0,1],s=28,c=orange,marker="x",linewidths=1.6,label="classe 0")
    ax.scatter(X_nl[y_nl==1,0],X_nl[y_nl==1,1],s=28,facecolors="white",edgecolors=blue,linewidths=1.3,label="classe 1")
    ax.set(title=title,xlabel=r"atributo $x_1$",ylabel=r"atributo $x_2$")
    ax.text(.03,.96,f"acurácia no exemplo: {(prediction==y_nl).mean():.2f}",transform=ax.transAxes,
            va="top",fontsize=12,bbox={"boxstyle":"round,pad=.25","facecolor":"white","alpha":.9,"edgecolor":"#b8c2cc"})
axes[1].legend(loc="lower right",fontsize=10)
save(fig,"fronteira-nao-linear.png")

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

# Complete example with a stratified train/test split.
rng=np.random.default_rng(20261111)
test_parts=[]
for label in [0,1]:
    ids=np.flatnonzero(y==label)
    rng.shuffle(ids)
    test_parts.append(ids[:round(.20*len(ids))])
test_ids=np.concatenate(test_parts); rng.shuffle(test_ids)
train_ids=np.setdiff1d(np.arange(len(y)),test_ids)

Xraw=df[cols].to_numpy(float)
train_mean=Xraw[train_ids].mean(axis=0); train_sd=Xraw[train_ids].std(axis=0)
Xtrain=np.column_stack([np.ones(len(train_ids)),(Xraw[train_ids]-train_mean)/train_sd])
Xtest=np.column_stack([np.ones(len(test_ids)),(Xraw[test_ids]-train_mean)/train_sd])
ytrain=y[train_ids]; ytest=y[test_ids]

b_train=np.zeros(Xtrain.shape[1]); hist_train=[]
for t in range(3000):
    p_train=sig(Xtrain@b_train)
    loss=-np.mean(ytrain*np.log(p_train+1e-12)+(1-ytrain)*np.log(1-p_train+1e-12))
    gradient=Xtrain.T@(p_train-ytrain)/len(ytrain)
    hist_train.append(loss)
    b_train-=.15*gradient

ptest=sig(Xtest@b_train); predtest=ptest>=.5
tp=int(np.sum(predtest&(ytest==1))); fn=int(np.sum(~predtest&(ytest==1)))
tn=int(np.sum(~predtest&(ytest==0))); fp=int(np.sum(predtest&(ytest==0)))
accuracy=(tp+tn)/len(ytest); sensitivity=tp/(tp+fn); specificity=tn/(tn+fp)
precision=tp/(tp+fp); f1=2*precision*sensitivity/(precision+sensitivity)

fig,axes=plt.subplots(1,2,figsize=(12,4.5))
axes[0].semilogy(hist_train,lw=3,color=blue)
axes[0].set(xlabel="iteração",ylabel="entropia cruzada no treino",title="Ajuste apenas no conjunto de treino")
test_probabilities=pd.DataFrame({"probabilidade":ptest,"classe":np.where(ytest==1,"fumante","não fumante")})
sns.histplot(data=test_probabilities,x="probabilidade",hue="classe",bins=18,element="step",stat="density",common_norm=False,ax=axes[1])
axes[1].axvline(.5,ls="--",color="black")
axes[1].set(xlabel="probabilidade prevista no teste",title="Classes avaliadas em dados separados")
save(fig,"exemplo-ajuste-teste.png")

fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
confusion=np.array([[tn,fp],[fn,tp]])
sns.heatmap(confusion,annot=True,fmt="d",cmap="Blues",cbar=False,ax=axes[0],
            xticklabels=["previsto 0","previsto 1"],yticklabels=["real 0","real 1"],annot_kws={"fontsize":18})
axes[0].set(title="Matriz de confusão no teste",xlabel="",ylabel="")
metric_names=["acurácia","sensibilidade","especificidade","precisão"]
metric_values=[accuracy,sensitivity,specificity,precision]
axes[1].barh(metric_names,metric_values,color=[blue,orange,green,"#6a4c93"])
axes[1].set(xlim=(0,1),xlabel="proporção",title="Cada métrica responde a uma pergunta")
for j,value in enumerate(metric_values): axes[1].text(value+.015,j,f"{value:.2f}",va="center",fontsize=13)
save(fig,"exemplo-confusao-metricas.png")

test_thresholds=np.linspace(.05,.95,37); test_sens=[]; test_spec=[]; test_precision=[]
for th in test_thresholds:
    q=ptest>=th
    th_tp=np.sum(q&(ytest==1)); th_fn=np.sum(~q&(ytest==1))
    th_tn=np.sum(~q&(ytest==0)); th_fp=np.sum(q&(ytest==0))
    test_sens.append(th_tp/(th_tp+th_fn))
    test_spec.append(th_tn/(th_tn+th_fp))
    test_precision.append(th_tp/(th_tp+th_fp) if th_tp+th_fp else np.nan)

fig,ax=plt.subplots(figsize=(9.5,5.1))
metric_curves=[("sensibilidade",test_sens,orange),("especificidade",test_spec,green),("precisão",test_precision,blue)]
for name,values,color in metric_curves:
    ax.plot(test_thresholds,values,lw=3,label=name,color=color)
reference_index=int(np.argmin(np.abs(test_thresholds-.5)))
reference_threshold=test_thresholds[reference_index]
ax.axvline(reference_threshold,ls="--",color="black",label=r"limiar $c=0{,}5$")
annotation_offsets={"sensibilidade":(-82,-30),"especificidade":(12,12),"precisão":(12,-24)}
for name,values,color in metric_curves:
    value=values[reference_index]
    ax.scatter([reference_threshold],[value],s=70,color=color,zorder=4,edgecolor="white",linewidth=1)
    ax.annotate(f"{name}: {value:.2f}",xy=(reference_threshold,value),
                xytext=annotation_offsets[name],textcoords="offset points",fontsize=11,color=color,
                arrowprops={"arrowstyle":"-","color":color,"lw":1.2})
ax.set(xlabel="limiar de classificação",ylabel="proporção",title="Mudar o limiar altera quais erros aceitamos")
ax.legend(fontsize=10,loc="lower left")
save(fig,"exemplo-limiar.png")

test_cal=pd.DataFrame({"p":ptest,"y":ytest})
test_cal["bin"]=pd.qcut(test_cal.p.rank(method="first"),6)
test_cal=test_cal.groupby("bin",observed=True).agg(observed=("y","mean"),predicted=("p","mean"),n=("y","size")).reset_index(drop=True)
test_cal["group"]=[f"G{i}" for i in range(1,len(test_cal)+1)]
fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={"width_ratios":[1.35,1]})
axes[0].plot([0,1],[0,1],"--",color="black",label="calibração ideal")
axes[0].scatter(test_cal.predicted,test_cal.observed,s=test_cal.n*5,color=blue,alpha=.8)
for row in test_cal.iloc[4:].itertuples():
    axes[0].annotate(row.group,(row.predicted,row.observed),xytext=(8,8),textcoords="offset points",fontsize=11,fontweight="bold")
axes[0].annotate("G1–G4 sobrepostos",xy=(test_cal.predicted.iloc[:4].mean(),0),xytext=(.08,.13),
                 arrowprops={"arrowstyle":"->","color":"#4c5967"},fontsize=11,color="#4c5967")
axes[0].set(xlim=(0,1),ylim=(0,1),xlabel="probabilidade média prevista",ylabel="proporção observada de fumantes",title="Visão completa")
axes[1].plot([0,.025],[0,.025],"--",color="black")
low_groups=test_cal.iloc[:4]
axes[1].scatter(low_groups.predicted,low_groups.observed,s=low_groups.n*5,color=blue,alpha=.8)
for index,row in enumerate(low_groups.itertuples()):
    axes[1].annotate(row.group,(row.predicted,row.observed),xytext=(0,12+12*(index%2)),
                     textcoords="offset points",ha="center",fontsize=11,fontweight="bold",
                     arrowprops={"arrowstyle":"-","color":blue,"lw":1})
axes[1].set(xlim=(-.001,.022),ylim=(-.002,.03),xlabel="probabilidade média prevista",ylabel="proporção observada",title="Zoom nos grupos de baixa probabilidade")
save(fig,"exemplo-calibracao.png")

print({"beta":b,"accuracy":(pred==y).mean(),"prevalence":y.mean()})
