from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
OUT = ROOT / "slides" / "assets" / "aula04-eda-olist"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0f6b78"
ORANGE = "#d95f02"
GREEN = "#3a7d44"
RED = "#b23b3b"
GRAY = "#6c737b"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 160, "axes.titleweight": "bold"})

orders = pd.read_csv(DATA / "orders.csv", parse_dates=[
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
])
customers = pd.read_csv(DATA / "customers.csv")
items = pd.read_csv(DATA / "order_items.csv")
reviews = pd.read_csv(DATA / "order_reviews.csv")

base = orders.merge(customers, on="customer_id", validate="many_to_one")
item_agg = items.groupby("order_id").agg(
    itens=("order_item_id", "size"),
    valor_produtos=("price", "sum"),
    frete=("freight_value", "sum"),
).reset_index()
review_agg = reviews.groupby("order_id").agg(
    nota=("review_score", "mean"),
    n_reviews=("review_id", "size"),
).reset_index()
analysis = (base.merge(item_agg, on="order_id", how="left", validate="one_to_one")
            .merge(review_agg, on="order_id", how="left", validate="one_to_one"))
analysis["entrega_dias"] = (
    analysis["order_delivered_customer_date"] - analysis["order_purchase_timestamp"]
).dt.total_seconds() / 86400
analysis["atraso_dias"] = (
    analysis["order_delivered_customer_date"] - analysis["order_estimated_delivery_date"]
).dt.total_seconds() / 86400
analysis["atrasou"] = analysis["atraso_dias"].gt(0)


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()


status = orders["order_status"].value_counts().sort_values()
plt.figure(figsize=(10, 5.2))
colors = [GRAY] * len(status)
colors[-1] = BLUE
status.plot.barh(color=colors)
plt.title("Pedidos entregues dominam a amostra")
plt.xlabel("pedidos")
plt.ylabel("")
save("status-pedidos.png")

missing = orders.isna().sum().sort_values()
missing = missing[missing > 0]
plt.figure(figsize=(10, 4.8))
missing.plot.barh(color=[ORANGE, ORANGE, RED])
plt.title("Ausências acompanham o estágio do pedido")
plt.xlabel("valores ausentes")
plt.ylabel("")
save("ausencias-pedidos.png")

monthly = orders.set_index("order_purchase_timestamp").resample("MS").size()
plt.figure(figsize=(11, 4.8))
monthly.plot(color=BLUE, linewidth=2.8)
plt.fill_between(monthly.index, monthly.values, color=BLUE, alpha=.14)
plt.title("A atividade cresce e termina com meses incompletos")
plt.xlabel("")
plt.ylabel("pedidos por mês")
save("pedidos-mensais.png")

per_order = items.groupby("order_id").size()
plt.figure(figsize=(10, 4.8))
sns.countplot(x=per_order.clip(upper=6), color=GREEN)
plt.title("A maioria dos pedidos possui apenas um item")
plt.xlabel("itens no pedido (6 = seis ou mais)")
plt.ylabel("pedidos")
save("itens-por-pedido.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
sns.histplot(analysis["valor_produtos"].dropna(), bins=70, color=BLUE, ax=axes[0])
axes[0].set_title("Escala completa")
axes[0].set_xlabel("valor dos produtos (R$)")
sns.histplot(analysis.loc[analysis["valor_produtos"].between(0, 500), "valor_produtos"],
             bins=50, color=BLUE, ax=axes[1])
axes[1].axvline(analysis["valor_produtos"].median(), color=ORANGE,
                linestyle="--", label=f"mediana = R$ {analysis['valor_produtos'].median():.0f}")
axes[1].legend(frameon=False)
axes[1].set_title("Zoom até R$ 500")
axes[1].set_xlabel("valor dos produtos (R$)")
fig.suptitle("A cauda longa esconde o pedido típico", fontweight="bold")
save("valor-pedidos.png")

delivered = analysis.query("order_status == 'delivered'").copy()
plt.figure(figsize=(10, 4.8))
sns.histplot(delivered["entrega_dias"].dropna(), bins=55, color=BLUE)
plt.axvline(delivered["entrega_dias"].median(), color=ORANGE, linestyle="--",
            label=f"mediana = {delivered['entrega_dias'].median():.1f} dias")
plt.legend(frameon=False)
plt.title("O tempo de entrega é assimétrico")
plt.xlabel("dias entre compra e entrega")
plt.ylabel("pedidos")
save("tempo-entrega.png")

score = reviews["review_score"].value_counts().sort_index()
plt.figure(figsize=(9, 4.8))
score.plot.bar(color=[RED, RED, ORANGE, BLUE, GREEN], rot=0)
plt.title("Notas máximas são frequentes")
plt.xlabel("nota da avaliação")
plt.ylabel("avaliações")
save("notas-avaliacoes.png")

late_score = delivered.groupby("atrasou")["nota"].agg(["count", "mean"]).reset_index()
late_score["prazo"] = late_score["atrasou"].map({False: "No prazo", True: "Atrasado"})
plt.figure(figsize=(8.8, 4.8))
ax = sns.barplot(data=late_score, x="prazo", y="mean", hue="prazo",
                 palette=[GREEN, RED], legend=False)
ax.set_ylim(0, 5)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=4)
plt.title("Atrasos aparecem junto de avaliações piores")
plt.xlabel("")
plt.ylabel("nota média")
save("nota-atraso.png")

states = analysis["customer_state"].value_counts().head(10).sort_values()
plt.figure(figsize=(9.5, 5.2))
states.plot.barh(color=BLUE)
plt.title("São Paulo concentra grande parte dos pedidos")
plt.xlabel("pedidos")
plt.ylabel("UF")
save("pedidos-estado.png")

print({
    "orders": orders.shape,
    "customers": customers.shape,
    "items": items.shape,
    "reviews": reviews.shape,
    "analysis": analysis.shape,
    "median_value": round(analysis["valor_produtos"].median(), 2),
    "median_delivery": round(delivered["entrega_dias"].median(), 2),
    "late_rate": round(delivered["atrasou"].mean(), 4),
    "score_on_time": round(late_score.loc[~late_score["atrasou"], "mean"].iloc[0], 3),
    "score_late": round(late_score.loc[late_score["atrasou"], "mean"].iloc[0], 3),
})
