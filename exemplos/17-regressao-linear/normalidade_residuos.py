"""QQ-plots e Anderson-Darling exploratório dos quatro ajustes."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad

from quatro_retas import ROOT, OUT, ajustar_grupos


def diagnosticar_normalidade(dados, resultados):
    linhas_figura = len(resultados) // 2
    fig, axes = plt.subplots(linhas_figura, 2, figsize=(11, 3.75 * linhas_figura), layout="constrained")
    linhas = []
    for ax, (_, row) in zip(axes.flat, resultados.iterrows()):
        smoker = "yes" if row.grupo == "Fumantes" else "no"
        grupo = dados[dados.smoker.eq(smoker) & dados.faixa_imc.eq(row.faixa_imc)]
        residuos = grupo.charges.to_numpy() - (row.b0 + row.b1*grupo.age.to_numpy())
        z = (residuos - residuos.mean()) / residuos.std(ddof=1)
        teoricos, observados = stats.probplot(z, dist="norm", fit=False)
        ax.scatter(teoricos, observados, s=20, alpha=.65, color="#0f6b78")
        limites = [min(teoricos), max(teoricos)]
        ax.plot(limites, limites, color="#d95f02", lw=2)
        ax.set(title=f"{row.grupo} · {row.faixa_imc} (n={row.n})",
               xlabel="Quantis teóricos da Normal", ylabel="Resíduos padronizados")
        ax.grid(alpha=.25)
        estatistica, p = normal_ad(residuos)
        linhas.append({"grupo": row.grupo, "faixa_imc": row.faixa_imc, "n": row.n,
                       "A2": float(estatistica), "valor_p": float(p)})
    return fig, pd.DataFrame(linhas)


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "exemplos/16-correlacao/data/insurance.csv")
    dados, resultados = ajustar_grupos(df)
    fig, testes = diagnosticar_normalidade(dados, resultados)
    fig.savefig(OUT / "qq-quatro-retas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for nome, arquivo in [("Fumantes", "qq-fumantes.png"),
                          ("Não fumantes", "qq-nao-fumantes.png")]:
        fig, _ = diagnosticar_normalidade(dados, resultados[resultados.grupo.eq(nome)])
        fig.savefig(OUT / arquivo, dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(testes.to_string(index=False))
