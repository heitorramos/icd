"""Resíduos versus valores ajustados dos quatro modelos da Aula 17."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from quatro_retas import OUT, ROOT, ajustar_grupos


CORES = ["#d95f02", "#a33d50", "#0f6b78", "#6a4c93"]


def diagnosticar_residuos(dados, resultados):
    """Constrói painéis comparáveis de resíduos versus valores ajustados."""
    paineis = []
    for _, row in resultados.iterrows():
        smoker = "yes" if row.grupo == "Fumantes" else "no"
        grupo = dados[
            dados.smoker.eq(smoker) & dados.faixa_imc.eq(row.faixa_imc)
        ]
        ajustados = row.b0 + row.b1 * grupo.age.to_numpy()
        residuos = grupo.charges.to_numpy() - ajustados
        paineis.append((row, ajustados, residuos))

    limite = 1.04 * max(np.abs(residuos).max() for _, _, residuos in paineis)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7.7), sharey=True, layout="constrained"
    )

    for ax, (row, ajustados, residuos), cor in zip(axes.flat, paineis, CORES):
        ax.scatter(
            ajustados,
            residuos,
            s=24,
            alpha=0.48,
            color=cor,
            edgecolor="none",
        )
        ax.axhline(0, color="#172b3a", linewidth=1.6, linestyle="--")

        resumo = pd.DataFrame(
            {"ajustado": ajustados, "residuo": residuos}
        )
        resumo["faixa"] = pd.qcut(
            resumo.ajustado, q=min(8, len(resumo)), duplicates="drop"
        )
        tendencia = resumo.groupby("faixa", observed=True).agg(
            ajustado=("ajustado", "mean"),
            residuo=("residuo", "mean"),
        )
        ax.plot(
            tendencia.ajustado,
            tendencia.residuo,
            color="#f2b134",
            linewidth=2.5,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label="Média em faixas",
        )

        rmse = f"{row.RMSE:,.0f}".replace(",", ".")
        ax.set_title(
            f"{row.grupo} · {row.faixa_imc}\n"
            f"n = {row.n} · RMSE = {rmse} US$",
            fontsize=12,
            fontweight="bold",
        )
        ax.set(
            xlabel="Valor ajustado (US$)",
            ylabel="Resíduo (US$)",
            ylim=(-limite, limite),
        )
        ax.tick_params(labelleft=True)
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.suptitle(
        "Resíduos dos quatro modelos: ainda há estrutura não explicada",
        fontsize=17,
        fontweight="bold",
    )
    return fig


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "exemplos/16-correlacao/data/insurance.csv")
    dados, resultados = ajustar_grupos(df)
    fig = diagnosticar_residuos(dados, resultados)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / "residuos-quatro-modelos.png",
        dpi=190,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
