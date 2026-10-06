#!/usr/bin/env python3
"""Sincroniza no cronograma as aulas liberadas na página de Aulas.

Uma aula é considerada liberada quando seu card em ``aulas/index.qmd`` contém
um link ``Abrir a aula``. O cronograma passa a receber automaticamente o link
correspondente na coluna Material antes de cada renderização do site.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AULAS_INDEX = ROOT / "aulas" / "index.qmd"
CRONOGRAMA = ROOT / "cronograma.qmd"

CARD_RE = re.compile(
    r"^### Aula\s+(\d{1,2})\s*$.*?(?=^### Aula\s+\d{1,2}\s*$|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
LINK_RE = re.compile(r"\[Abrir a aula\]\(([^)]+\.qmd)\)")


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


def main() -> None:
    liberadas = aulas_liberadas(AULAS_INDEX.read_text(encoding="utf-8"))
    original = CRONOGRAMA.read_text(encoding="utf-8")
    atualizado = sincronizar_cronograma(original, liberadas)

    if atualizado != original:
        CRONOGRAMA.write_text(atualizado, encoding="utf-8")
        numeros = ", ".join(f"{numero:02d}" for numero in sorted(liberadas))
        print(f"Cronograma sincronizado com as aulas liberadas: {numeros}")
    else:
        print("Cronograma já está sincronizado com as aulas liberadas.")


if __name__ == "__main__":
    main()
