#!/usr/bin/env python3
"""Sincroniza cronograma e menu lateral com as aulas liberadas.

Uma aula é considerada liberada quando seu card em ``aulas/index.qmd`` contém
um link ``Abrir a aula``. Antes de cada renderização, o cronograma recebe o link
correspondente e o menu lateral passa a incluir o hotsite e a associação interna
com seu material de apoio.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AULAS_INDEX = ROOT / "aulas" / "index.qmd"
CRONOGRAMA = ROOT / "cronograma.qmd"
QUARTO = ROOT / "_quarto.yml"

CARD_RE = re.compile(
    r"^### Aula\s+(\d{1,2})\s*$.*?(?=^### Aula\s+\d{1,2}\s*$|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
LINK_RE = re.compile(r"\[Abrir a aula\]\(([^)]+\.qmd)\)")
MATERIAL_RE = re.compile(
    r"\]\(\.\./(exemplos/[^)]+/notebook\.(?:qmd|ipynb))\)"
)
SIDEBAR_START = "        # BEGIN AUTO: aulas liberadas\n"
SIDEBAR_END = "        # END AUTO: aulas liberadas\n"
MODULOS = (
    ("Módulo 0 · Motivação", range(1, 3)),
    ("Módulo 1 · Análise Exploratória", range(3, 8)),
    ("Módulo 2 · Inferência Estatística", range(8, 16)),
    ("Módulo 3 · Modelagem e Aprendizado de Máquina", range(16, 26)),
)


def aulas_liberadas(texto: str) -> dict[int, str]:
    """Retorna ``{numero: caminho}`` para os cards com acesso liberado."""
    liberadas: dict[int, str] = {}
    for card in CARD_RE.finditer(texto):
        numero = int(card.group(1))
        link = LINK_RE.search(card.group(0))
        if link:
            liberadas[numero] = link.group(1)
    return liberadas


def sincronizar_cronograma(texto: str, liberadas: dict[int, str]) -> str:
    """Insere os links liberados nas linhas correspondentes do cronograma."""
    linhas_atualizadas: list[str] = []
    encontradas: set[int] = set()

    for linha in texto.splitlines(keepends=True):
        quebra = "\n" if linha.endswith("\n") else ""
        conteudo = linha[:-1] if quebra else linha
        celulas = conteudo.split("|")

        if len(celulas) >= 6 and celulas[1].strip().isdigit():
            numero = int(celulas[1].strip())
            if numero in liberadas:
                alvo = liberadas[numero]
                if not (ROOT / "aulas" / alvo).is_file():
                    raise FileNotFoundError(
                        f"Aula {numero:02d} liberada, mas o arquivo não existe: "
                        f"aulas/{alvo}"
                    )
                celulas[4] = f" [Acessar](aulas/{alvo}) "
                conteudo = "|".join(celulas)
                encontradas.add(numero)

        linhas_atualizadas.append(conteudo + quebra)

    ausentes = sorted(set(liberadas) - encontradas)
    if ausentes:
        lista = ", ".join(f"Aula {numero:02d}" for numero in ausentes)
        raise ValueError(f"Aulas liberadas sem linha no cronograma: {lista}")

    return "".join(linhas_atualizadas)


def material_de_apoio(numero: int, aula: str) -> str:
    """Encontra e valida o material de apoio vinculado ao hotsite da aula."""
    arquivo_aula = ROOT / "aulas" / aula
    match = MATERIAL_RE.search(arquivo_aula.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(
            f"Aula {numero:02d} liberada sem link para material de apoio: "
            f"aulas/{aula}"
        )

    material = match.group(1)
    if not (ROOT / material).is_file():
        raise FileNotFoundError(
            f"Material de apoio da Aula {numero:02d} não existe: {material}"
        )
    return material


def bloco_sidebar(liberadas: dict[int, str]) -> str:
    """Gera os módulos do menu a partir dos cards efetivamente liberados."""
    cobertas = {numero for _, numeros in MODULOS for numero in numeros}
    fora_dos_modulos = sorted(set(liberadas) - cobertas)
    if fora_dos_modulos:
        lista = ", ".join(f"Aula {numero:02d}" for numero in fora_dos_modulos)
        raise ValueError(f"Aulas liberadas sem módulo configurado: {lista}")

    linhas: list[str] = []
    materiais_incluidos: set[str] = set()

    for titulo, numeros in MODULOS:
        aulas_do_modulo = [numero for numero in numeros if numero in liberadas]
        if not aulas_do_modulo:
            continue

        linhas.append(f'        - section: "{titulo}"\n')
        linhas.append("          contents:\n")

        for numero in aulas_do_modulo:
            aula = liberadas[numero]
            arquivo_aula = ROOT / "aulas" / aula
            if not arquivo_aula.is_file():
                raise FileNotFoundError(
                    f"Aula {numero:02d} liberada, mas o arquivo não existe: "
                    f"aulas/{aula}"
                )

            linhas.append(f"            - aulas/{aula}\n")
            material = material_de_apoio(numero, aula)
            if material not in materiais_incluidos:
                linhas.append(
                    f'            - text: "Material de apoio · Aula {numero:02d}"\n'
                )
                linhas.append(f"              href: {material}\n")
                materiais_incluidos.add(material)

    return "".join(linhas)


def sincronizar_sidebar(texto: str, liberadas: dict[int, str]) -> str:
    """Substitui o bloco automático do menu pelas aulas liberadas."""
    inicio = texto.find(SIDEBAR_START)
    fim = texto.find(SIDEBAR_END)
    if inicio == -1 or fim == -1 or fim < inicio:
        raise ValueError("Marcadores automáticos do menu lateral não encontrados.")

    inicio_conteudo = inicio + len(SIDEBAR_START)
    return (
        texto[:inicio_conteudo]
        + bloco_sidebar(liberadas)
        + texto[fim:]
    )


def main() -> None:
    liberadas = aulas_liberadas(AULAS_INDEX.read_text(encoding="utf-8"))
    cronograma_original = CRONOGRAMA.read_text(encoding="utf-8")
    cronograma_atualizado = sincronizar_cronograma(cronograma_original, liberadas)
    quarto_original = QUARTO.read_text(encoding="utf-8")
    quarto_atualizado = sincronizar_sidebar(quarto_original, liberadas)

    alterados: list[str] = []
    if cronograma_atualizado != cronograma_original:
        CRONOGRAMA.write_text(cronograma_atualizado, encoding="utf-8")
        alterados.append("cronograma")
    if quarto_atualizado != quarto_original:
        QUARTO.write_text(quarto_atualizado, encoding="utf-8")
        alterados.append("menu lateral")

    numeros = ", ".join(f"{numero:02d}" for numero in sorted(liberadas))
    if alterados:
        itens = " e ".join(alterados)
        print(f"Sincronização atualizada ({itens}) para as aulas: {numeros}")
    else:
        print(f"Cronograma e menu lateral já sincronizados com as aulas: {numeros}")


if __name__ == "__main__":
    main()
