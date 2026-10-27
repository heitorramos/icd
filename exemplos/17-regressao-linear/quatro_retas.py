"""Quatro ajustes por tabagismo e faixa de IMC, sem selecionar pela resposta."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides/assets/aula17-regressao-linear"


def ajustar_grupos(df):
    dados = df.copy()
    if dados[["age", "charges", "bmi", "smoker"]].isna().any().any():
        raise ValueError("Verifique os valores ausentes antes de definir os grupos.")
    dados["faixa_imc"] = np.where(dados.bmi < 30, "IMC < 30", "IMC ≥ 30")
    resultados = []
    for smoker, nome in [("yes", "Fumantes"), ("no", "Não fumantes")]:
        for faixa_imc in ["IMC < 30", "IMC ≥ 30"]:
            grupo = dados[dados.smoker.eq(smoker) & dados.faixa_imc.eq(faixa_imc)]
            x, y = grupo.age.to_numpy(), grupo.charges.to_numpy()
            b1 = np.sum((x-x.mean())*(y-y.mean())) / np.sum((x-x.mean())**2)
            b0 = y.mean() - b1*x.mean()
            residuos = y - (b0 + b1*x)
            resultados.append({"grupo": nome, "faixa_imc": faixa_imc, "n": len(grupo),
                               "b0": b0, "b1": b1,
                               "R2": 1 - np.sum(residuos**2)/np.sum((y-y.mean())**2),
                               "RMSE": np.sqrt(np.mean(residuos**2))})
    return dados, pd.DataFrame(resultados)


def desenhar_grupos(dados, resultados):
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
    linhas = len(resultados) // 2
    fig, axes = plt.subplots(linhas, 2, figsize=(12, 4 * linhas), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (_, row), cor in zip(axes.flat, resultados.iterrows(),
                                 ["#d95f02", "#a33d50", "#0f6b78", "#6a4c93"]):
        smoker = "yes" if row.grupo == "Fumantes" else "no"
        grupo = dados[dados.smoker.eq(smoker) & dados.faixa_imc.eq(row.faixa_imc)]
        ax.scatter(grupo.age, grupo.charges, s=17, alpha=.45, color=cor)
        xx = np.array([grupo.age.min(), grupo.age.max()])
        ax.plot(xx, row.b0 + row.b1*xx, color="#172b3a", lw=2.5)
        texto = f"n = {row.n} | R² = {row.R2:.3f} | RMSE = {row.RMSE:,.0f} US$"
        texto = texto.replace(",", "X").replace(".", ",").replace("X", ".")
        ax.set_title(f"{row.grupo} · {row.faixa_imc}\n{texto}", fontsize=12)
        ax.set(xlabel="Idade (anos)", ylabel="Despesas (US$)",
               xlim=(16, 66), ylim=(0, 68000))
        ax.tick_params(labelleft=True, labelbottom=True)
    fig.suptitle("Tabagismo e IMC: quatro ajustes nos próprios dados", fontweight="bold")
    return fig


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "exemplos/16-correlacao/data/insurance.csv")
    dados, resultados = ajustar_grupos(df)
    fig = desenhar_grupos(dados, resultados)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "quatro-retas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for nome, arquivo in [("Fumantes", "retas-fumantes.png"),
                          ("Não fumantes", "retas-nao-fumantes.png")]:
        fig = desenhar_grupos(dados, resultados[resultados.grupo.eq(nome)])
        fig.savefig(OUT / arquivo, dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(resultados.round(3).to_string(index=False))
