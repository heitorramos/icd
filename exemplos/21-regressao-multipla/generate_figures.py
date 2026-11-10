from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"slides/assets/aula21-regressao-multipla"; OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/"exemplos/16-correlacao/data/insurance.csv")
sns.set_theme(style="whitegrid",context="talk"); blue,orange,green="#1f6f8b","#d95f02","#2a9d8f"

def save(fig,name): fig.tight_layout(); fig.savefig(OUT/name,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.5)); sns.scatterplot(data=df,x="age",y="charges",hue="smoker",alpha=.45,ax=axes[0]); axes[0].set_title("Idade não conta toda a história"); sns.boxplot(data=df,x="region",y="charges",hue="smoker",ax=axes[1]); axes[1].tick_params(axis="x",rotation=20); axes[1].set_title("Grupos categóricos também entram no modelo"); save(fig,"descritiva-multipla.png")

# design matrix with reference categories
Xdf=pd.DataFrame({"age":df.age,"bmi":df.bmi,"children":df.children,"smoker_yes":(df.smoker=="yes").astype(int),"sex_male":(df.sex=="male").astype(int)})
Xdf=pd.concat([Xdf,pd.get_dummies(df.region,prefix="region",drop_first=True,dtype=int)],axis=1)
names=["intercepto"]+list(Xdf.columns); X=np.column_stack([np.ones(len(df)),Xdf.to_numpy(float)]); y=df.charges.to_numpy(float)
beta=np.linalg.lstsq(X,y,rcond=None)[0]; pred=X@beta; resid=y-pred
Xs=(Xdf-Xdf.mean())/Xdf.std(); Xstd=np.column_stack([np.ones(len(df)),Xs.to_numpy(float)]); bstd=np.linalg.lstsq(Xstd,y,rcond=None)[0]

# gradient descent convergence on the same multiple-regression example
Xgd=X.copy(); Xgd[:,1:]=(Xgd[:,1:]-Xgd[:,1:].mean(axis=0))/Xgd[:,1:].std(axis=0)
bgd=np.zeros(Xgd.shape[1]); eta_gd=.01; history=[]
for _ in range(50_000):
    residual_gd=y-Xgd@bgd
    history.append(np.mean(residual_gd**2))
    gradient_gd=-(2/len(y))*Xgd.T@residual_gd
    if np.linalg.norm(gradient_gd)<1e-6: break
    bgd=bgd-eta_gd*gradient_gd
history=np.asarray(history); mse_ols=np.mean(resid**2)
fig,ax=plt.subplots(figsize=(10,5.2)); ax.plot(np.arange(len(history)),history/1e6,color=blue,lw=3,label="gradiente descendente"); ax.axhline(mse_ols/1e6,color=orange,ls="--",lw=2.5,label="mínimo por OLS"); ax.scatter(len(history)-1,history[-1]/1e6,s=65,color=blue,zorder=3); ax.set(xlabel="iteração",ylabel=r"MSE (milhões de US$²)",title="A perda cai e se aproxima do mínimo de OLS"); ax.legend(); save(fig,"convergencia-gd.png")

fig,ax=plt.subplots(figsize=(9,5)); order=np.argsort(bstd[1:]); ax.barh(np.array(names[1:])[order],bstd[1:][order],color=[orange if v<0 else blue for v in bstd[1:][order]]); ax.axvline(0,color="black",lw=1); ax.set(xlabel="mudança prevista em charges por 1 desvio-padrão",title="Coeficientes padronizados não medem causalidade"); save(fig,"coeficientes-padronizados.png")

simple=np.column_stack([np.ones(len(df)),df.age]); bs=np.linalg.lstsq(simple,y,rcond=None)[0]; ps=simple@bs; es=y-ps
fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True); axes[0].scatter(ps,es,s=10,alpha=.3,color=blue); axes[0].axhline(0,color="black",ls="--"); axes[0].set(title="Modelo simples",xlabel="ajustado",ylabel="resíduo"); axes[1].scatter(pred,resid,s=10,alpha=.3,color=green); axes[1].axhline(0,color="black",ls="--"); axes[1].set(title="Modelo múltiplo",xlabel="ajustado"); save(fig,"residuos-comparacao.png")

# visual diagnostics before and after the BMI-by-smoking interaction
Xinter=Xdf.copy(); Xinter["bmi_c"]=Xinter.pop("bmi")-30; Xinter["bmi_c_x_smoker"]=Xinter.bmi_c*Xinter.smoker_yes
Xi=np.column_stack([np.ones(len(df)),Xinter.to_numpy(float)]); bi=np.linalg.lstsq(Xi,y,rcond=None)[0]; predi=Xi@bi; residi=y-predi
probs=(np.arange(1,len(y)+1)-.5)/len(y); theoretical=np.array([NormalDist().inv_cdf(float(p)) for p in probs])
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex="col",sharey="col")
for row,(label,fitted,error,color) in enumerate([("Sem interação",pred,resid,blue),("Com IMC × tabagismo",predi,residi,orange)]):
    ordered=np.sort(error); slope,intercept=np.polyfit(theoretical,ordered,1)
    axes[row,0].scatter(theoretical,ordered,s=10,alpha=.45,color=color); axes[row,0].plot(theoretical,intercept+slope*theoretical,color="black",lw=1.5)
    axes[row,0].set(title=f"{label}: QQ-plot",ylabel="resíduos ordenados")
    axes[row,1].scatter(fitted,error,s=10,alpha=.28,color=color); axes[row,1].axhline(0,color="black",ls="--",lw=1.2)
    axes[row,1].set(title=f"{label}: resíduos × ajustados",ylabel="resíduo")
axes[1,0].set_xlabel("quantis teóricos"); axes[1,1].set_xlabel("valor ajustado"); save(fig,"diagnostico-interacao.png")

fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(y,pred,s=12,alpha=.35,color=blue); lo=min(y.min(),pred.min()); hi=max(y.max(),pred.max()); ax.plot([lo,hi],[lo,hi],ls="--",color=orange); ax.set(xlabel="observado",ylabel="ajustado",title="Previsões ainda têm grande dispersão"); save(fig,"observado-ajustado.png")

# bootstrap coefficients
rng=np.random.default_rng(20260827); B=1000; boots=np.empty((B,X.shape[1]))
for b in range(B):
    ii=rng.integers(0,len(df),len(df)); boots[b]=np.linalg.lstsq(X[ii],y[ii],rcond=None)[0]
sel=[1,2,3,4]; fig,ax=plt.subplots(figsize=(9,5));
for j,c in zip(sel,[blue,green,orange,"#6a3d9a"]): sns.kdeplot(boots[:,j],ax=ax,label=names[j],lw=2.5,color=c)
ax.set(xlabel="coeficiente bootstrap",ylabel="densidade",title="A escala e a precisão variam entre coeficientes"); ax.legend(); save(fig,"bootstrap-coeficientes.png")

# collinearity synthetic
rng=np.random.default_rng(3); z=rng.normal(size=200); x1=z+rng.normal(scale=.12,size=200); x2=z+rng.normal(scale=.12,size=200); yy=2+3*x1+rng.normal(size=200)
fig,axes=plt.subplots(1,2,figsize=(12,4.5)); axes[0].scatter(x1,x2,s=18,alpha=.5,color=blue); axes[0].set(xlabel="$x_1$",ylabel="$x_2$",title="Preditores quase redundantes");
coefs=[]
for _ in range(700):
    ii=rng.integers(0,len(z),len(z)); coefs.append(np.linalg.lstsq(np.column_stack([np.ones(len(ii)),x1[ii],x2[ii]]),yy[ii],rcond=None)[0][1:])
coefs=np.array(coefs); axes[1].scatter(coefs[:,0],coefs[:,1],s=10,alpha=.25,color=orange); axes[1].set(xlabel=r"$\hat\beta_1$",ylabel=r"$\hat\beta_2$",title="Coeficientes individuais ficam instáveis"); save(fig,"colinearidade.png")

print(pd.Series(beta,index=names).round(3).to_dict())
