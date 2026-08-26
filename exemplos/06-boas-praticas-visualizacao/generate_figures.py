from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(__file__).parent/"data"
OUT=ROOT/"slides/assets/aula06-visualizacao"; OUT.mkdir(parents=True,exist_ok=True)
BLUE,ORANGE,GREEN,RED,GRAY="#176b87","#d97727","#3a7d44","#b23b3b","#737b83"
sns.set_theme(style="whitegrid",context="talk"); plt.rcParams.update({"figure.dpi":150,"axes.titleweight":"bold"})
m=pd.read_csv(DATA/"tmdb_movies.csv")
m["release_date"]=pd.to_datetime(m.release_date,errors="coerce")
m["year"]=m.release_date.dt.year
m["profit"]=m.revenue-m.budget
m["roi"]=m.revenue/m.budget
def genre(x):
    try: return json.loads(x)[0]["name"] if json.loads(x) else "Sem gênero"
    except Exception: return "Sem gênero"
m["main_genre"]=m.genres.map(genre)
valid=m.query("budget > 100000 and revenue > 100000").copy()
olist_orders=pd.read_csv(DATA/"olist_orders.csv",parse_dates=["order_purchase_timestamp"])
olist_customers=pd.read_csv(DATA/"olist_customers.csv")
olist=olist_orders.merge(olist_customers,on="customer_id",validate="many_to_one")
def save(name): plt.tight_layout(); plt.savefig(OUT/name,bbox_inches="tight",facecolor="white"); plt.close()

top=m.main_genre.value_counts().head(8).sort_values()
plt.figure(figsize=(9.5,5)); top.plot.barh(color=BLUE); plt.title("Barras ordenadas facilitam a comparação"); plt.xlabel("filmes"); plt.ylabel(""); save("generos-ordenados.png")

pair=m.main_genre.value_counts().loc[["Drama", "Comedy"]]
pair.index=["Drama","Comédia"]
fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True)
pair.plot.bar(color=[BLUE,ORANGE],rot=0,ax=axes[0]); axes[0].set_ylim(0,pair.max()*1.12); axes[0].set_title("Eixo começa em zero"); axes[0].set_ylabel("filmes")
pair.plot.bar(color=[BLUE,ORANGE],rot=0,ax=axes[1]); axes[1].set_ylim(pair.min()*.94,pair.max()*1.02); axes[1].set_title("Eixo truncado exagera a diferença"); axes[1].set_ylabel("filmes")
for ax in axes: ax.set_xlabel("")
fig.suptitle("A escala muda a história visual",fontweight="bold"); save("eixo-truncado.png")

fig,axes=plt.subplots(1,2,figsize=(12,4.8))
sns.scatterplot(data=valid,x="budget",y="revenue",alpha=.35,color=GRAY,ax=axes[0]); axes[0].set_title("Escala linear"); axes[0].set_xlabel("orçamento"); axes[0].set_ylabel("receita")
sns.scatterplot(data=valid,x="budget",y="revenue",alpha=.35,color=BLUE,ax=axes[1]); axes[1].set(xscale="log",yscale="log",title="Escala log-log",xlabel="orçamento",ylabel="receita")
save("linear-log.png")

genres=top.index[-5:]
d=m[m.main_genre.isin(genres)&m.vote_average.notna()]
plt.figure(figsize=(10,5)); sns.boxplot(data=d,x="vote_average",y="main_genre",order=d.groupby("main_genre").vote_average.median().sort_values().index,color=BLUE)
plt.title("Distribuições contam mais que médias"); plt.xlabel("nota média"); plt.ylabel(""); save("notas-genero-box.png")

annual=(m.query("1980 <= year <= 2016").groupby("year").agg(filmes=("id","size"),nota=("vote_average","mean")).reset_index())

fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,invert,title in zip(axes,[False,True],["Eixo usual: curvas se opõem","Eixo invertido: curvas acompanham"]):
    ax2=ax.twinx()
    ax.plot(annual.year,annual.filmes,color=BLUE,linewidth=2.6)
    ax2.plot(annual.year,annual.nota,color=ORANGE,linewidth=2.6)
    if invert: ax2.invert_yaxis()
    ax.set_title(title); ax.set_xlabel("ano")
    ax.set_ylabel("filmes",color=BLUE); ax2.set_ylabel("nota média",color=ORANGE)
    ax.tick_params(axis="y",colors=BLUE); ax2.tick_params(axis="y",colors=ORANGE)
fig.suptitle("Mesmos dados, impressão oposta: o segundo eixo é arbitrário",fontweight="bold")
save("eixos-duplos.png")

plt.figure(figsize=(10.5,5)); sns.lineplot(data=annual,x="year",y="filmes",color=BLUE,linewidth=2.6); plt.title("Tempo pede ordem e continuidade"); plt.xlabel(""); plt.ylabel("filmes na base"); save("filmes-tempo.png")

annual_genres=(m.query("1980 <= year <= 2016 and main_genre in ['Drama','Comedy']")
               .groupby(["year","main_genre"]).size().rename("filmes").reset_index())
ratio=annual_genres.pivot(index="year",columns="main_genre",values="filmes").fillna(0)
ratio["Drama / Comédia"]=ratio["Drama"]/ratio["Comedy"].replace(0,np.nan)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
sns.lineplot(data=annual_genres,x="year",y="filmes",hue="main_genre",palette=[ORANGE,BLUE],ax=axes[0])
axes[0].set_title("Contagens absolutas"); axes[0].set_xlabel(""); axes[0].legend(title="gênero",frameon=False)
axes[1].plot(ratio.index,ratio["Drama / Comédia"],color=RED,linewidth=2.4)
axes[1].axhline(1,color=GRAY,linestyle="--"); axes[1].set_title("Razão Drama / Comédia"); axes[1].set_xlabel(""); axes[1].set_ylabel("razão")
fig.suptitle("Contagem e razão respondem perguntas diferentes",fontweight="bold"); save("contagem-razao.png")

top4=["Drama","Comedy","Action","Adventure"]
annual4=(m.query("1980 <= year <= 2016 and main_genre in @top4")
         .groupby(["year","main_genre"]).size().rename("filmes").reset_index())
g=sns.FacetGrid(annual4,col="main_genre",col_wrap=4,height=2.8,aspect=.95,sharex=True,sharey=True)
g.map_dataframe(sns.lineplot,x="year",y="filmes",color=BLUE,linewidth=2.2)
g.set_titles("{col_name}"); g.set_axis_labels("ano","filmes")
g.figure.suptitle("Mesmo domínio, lado a lado",y=1.05,fontweight="bold")
g.savefig(OUT/"justaposicao-generos.png",bbox_inches="tight",facecolor="white"); plt.close(g.figure)

geo=json.load(open(DATA/"brazil-states.geojson"))
quarter=olist[olist.order_purchase_timestamp.dt.year.eq(2017)].copy()
quarter["trimestre"]="T"+quarter.order_purchase_timestamp.dt.quarter.astype(str)
quarter_counts=quarter.groupby(["trimestre","customer_state"]).size().rename("pedidos")
vmax=quarter_counts.max()
fig,axes=plt.subplots(1,4,figsize=(13,4.5))
for ax,label in zip(axes,["T1","T2","T3","T4"]):
    values=quarter_counts.loc[label] if label in quarter_counts.index.levels[0] else pd.Series(dtype=float)
    patches=[]; colors=[]
    for feature in geo["features"]:
        uf=feature["properties"]["sigla"]
        for polygon in feature["geometry"]["coordinates"]:
            patches.append(Polygon(np.asarray(polygon[0]),closed=True)); colors.append(values.get(uf,0))
    coll=PatchCollection(
        patches,
        cmap="Blues",
        norm=Normalize(0,vmax),
        edgecolor="#34454f",
        linewidth=.8,
    )
    coll.set_array(np.asarray(colors)); ax.add_collection(coll); ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(label)
fig.colorbar(coll,ax=axes,orientation="horizontal",fraction=.06,pad=.06,label="pedidos na amostra")
fig.suptitle("Condicionando pedidos da Olist por espaço e trimestre — 2017",fontweight="bold")
plt.savefig(OUT/"olist-mapa-trimestres.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

best=(valid.query("vote_count >= 500").nlargest(10,"profit").sort_values("profit"))
plt.figure(figsize=(10,5.5)); plt.barh(best.original_title,best.profit/1e9,color=[GRAY]*9+[ORANGE]); plt.title("Destaque uma mensagem, não cada elemento"); plt.xlabel("lucro nominal (US$ bilhões)"); plt.ylabel(""); save("lucro-destaque.png")

rng=np.random.default_rng(42); categories=[f"Categoria {i}" for i in range(1,18)]; vals=np.sort(rng.integers(25,100,len(categories)))[::-1]
fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].bar(categories,vals,color=[plt.cm.tab20(i) for i in range(len(categories))]); axes[0].tick_params(axis="x",rotation=90); axes[0].set_title("Ruído")
axes[1].barh(categories[::-1],vals[::-1],color=[ORANGE if i==0 else "#b7c4ca" for i in range(len(vals))][::-1]); axes[1].set_title("Hierarquia visual")
save("ruido-clareza.png")

share=m.main_genre.value_counts().head(6)
fig,axes=plt.subplots(1,2,figsize=(11.5,5))
axes[0].pie(share,labels=share.index,autopct="%1.0f%%",startangle=90); axes[0].set_title("Ângulos")
share.sort_values().plot.barh(color=BLUE,ax=axes[1]); axes[1].set_title("Comprimentos alinhados"); axes[1].set_xlabel("filmes")
save("pizza-barras.png")

plt.figure(figsize=(10,5.2)); sc=plt.scatter(valid.budget,valid.revenue,c=np.log10(valid.vote_count.clip(lower=1)),cmap="viridis",alpha=.55,s=28); plt.xscale("log"); plt.yscale("log"); plt.colorbar(sc,label="log10(votos)"); plt.title("Cor sequencial representa magnitude"); plt.xlabel("orçamento"); plt.ylabel("receita"); save("cor-sequencial.png")

# Exemplos para contexto, anotação, rotulagem, aspecto, sobreposição,
# agregação e acessibilidade.
context=m.main_genre.value_counts().head(6).sort_values()
fig,ax=plt.subplots(figsize=(10.5,5.2))
context.plot.barh(ax=ax,color=BLUE)
ax.set_title("Drama e comédia dominam a amostra do TMDB 5000",loc="left")
ax.set_xlabel("número de filmes na base"); ax.set_ylabel("")
ax.text(1,-.2,"Unidade: filme · Base com 4.803 títulos · Fonte: TMDB 5000 / Kaggle",transform=ax.transAxes,fontsize=11,color=GRAY,ha="right")
save("contexto-completo.png")

fig,ax=plt.subplots(figsize=(10.5,5.2))
ax.plot(annual.year,annual.filmes,color=BLUE,linewidth=2.6)
peak=annual.loc[annual.filmes.idxmax()]
ax.scatter(peak.year,peak.filmes,color=ORANGE,s=75,zorder=3)
ax.annotate(f"Pico da base: {int(peak.filmes)} filmes em {int(peak.year)}",xy=(peak.year,peak.filmes),xytext=(-185,-55),textcoords="offset points",arrowprops={"arrowstyle":"->","color":ORANGE},color="#29343a")
ax.set(title="A anotação liga o padrão à interpretação",xlabel="ano de lançamento",ylabel="filmes na base")
save("anotacao-direta.png")

twogen=annual_genres[annual_genres.main_genre.isin(["Drama","Comedy"])]
fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True,sharey=True)
for genre,color in [("Drama",BLUE),("Comedy",ORANGE)]:
    d=twogen[twogen.main_genre.eq(genre)]
    axes[0].plot(d.year,d.filmes,label=genre,color=color,linewidth=2.4)
    axes[1].plot(d.year,d.filmes,color=color,linewidth=2.4)
    last=d.sort_values("year").iloc[-1]
    axes[1].text(last.year+.7,last.filmes,"Comédia" if genre=="Comedy" else genre,color=color,va="center")
axes[0].legend(title="gênero",frameon=False); axes[0].set_title("Legenda distante")
axes[1].set_title("Rótulo junto à série")
for ax in axes: ax.set_xlabel("ano"); ax.set_ylabel("filmes")
save("legenda-rotulo-direto.png")

fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={"width_ratios":[1,2.4]},sharey=True)
for ax in axes:
    ax.plot(annual.year,annual.filmes,color=BLUE,linewidth=2.5)
    ax.set_xlabel("ano"); ax.set_ylabel("filmes")
axes[0].set_title("Painel estreito"); axes[1].set_title("Painel largo")
fig.suptitle("Mesmos dados: a razão de aspecto muda a inclinação aparente",fontweight="bold")
save("razao-aspecto.png")

fig,axes=plt.subplots(1,2,figsize=(12,4.8))
axes[0].scatter(valid.budget,valid.revenue,s=22,alpha=1,color=BLUE)
axes[0].set_title("Pontos opacos se encobrem")
hb=axes[1].hexbin(valid.budget,valid.revenue,gridsize=32,mincnt=1,cmap="Blues",bins="log",xscale="log",yscale="log")
axes[1].set_title("Hexbin revela concentração")
for ax in axes: ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("orçamento"); ax.set_ylabel("receita")
fig.colorbar(hb,ax=axes[1],label="log10(contagem)")
save("sobreposicao-hexbin.png")

timed=m.query("1980 <= year <= 2016 and vote_average > 0").copy()
yearly=timed.groupby("year").vote_average.agg(mediana="median",q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index()
fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True,sharey=True)
axes[0].scatter(timed.year,timed.vote_average,s=10,alpha=.12,color=GRAY); axes[0].set_title("Cada filme")
axes[1].fill_between(yearly.year,yearly.q25,yearly.q75,color="#b8d6df",alpha=.8,label="50% central")
axes[1].plot(yearly.year,yearly.mediana,color=BLUE,linewidth=2.6,label="mediana anual"); axes[1].set_title("Resumo preserva tendência e dispersão"); axes[1].legend(frameon=False)
for ax in axes: ax.set_xlabel("ano"); ax.set_ylabel("nota")
save("agregacao-granularidade.png")

fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True,sharey=True)
for genre,color in [("Drama","#d64f4f"),("Comedy","#52a85d")]:
    d=twogen[twogen.main_genre.eq(genre)]
    axes[0].plot(d.year,d.filmes,color=color,linewidth=2.5,label=genre)
for genre,color,style in [("Drama",BLUE,"-"),("Comedy",ORANGE,"--")]:
    d=twogen[twogen.main_genre.eq(genre)]
    axes[1].plot(d.year,d.filmes,color=color,linestyle=style,linewidth=2.7,label="Comédia" if genre=="Comedy" else genre)
axes[0].set_title("Apenas cor"); axes[1].set_title("Cor + traço + rótulo")
for ax in axes: ax.legend(frameon=False); ax.set_xlabel("ano"); ax.set_ylabel("filmes")
save("acessibilidade-redundancia.png")

print(m.shape, valid.shape, m.main_genre.value_counts().head())
