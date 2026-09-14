#!/usr/bin/env python3
"""Generate Colab-ready notebooks and links for every published class material."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "heitorramos/icd"
BRANCH = "main"
NOTEBOOK_LINK = re.compile(
    r"\.\./(?P<path>exemplos/[^)\s]+/notebook\.(?:qmd|ipynb))"
)
BUTTON_BLOCK = re.compile(
    r"\n?<!-- colab-link:start -->.*?<!-- colab-link:end -->\n?",
    flags=re.DOTALL,
)


def colab_url(directory: Path) -> str:
    relative = directory.relative_to(ROOT).as_posix()
    return (
        "https://colab.research.google.com/github/"
        f"{REPOSITORY}/blob/{BRANCH}/{relative}/notebook-colab.ipynb"
    )


def tagged_cell(cell: dict, tag: str) -> bool:
    return tag in cell.get("metadata", {}).get("tags", [])


def prepare_notebook(notebook: dict, directory: Path) -> dict:
    relative = directory.relative_to(ROOT).as_posix()
    url = colab_url(directory)
    notebook["cells"] = [
        cell
        for cell in notebook.get("cells", [])
        if not tagged_cell(cell, "icd-colab-badge")
        and not tagged_cell(cell, "icd-colab-setup")
    ]

    badge = {
        "cell_type": "markdown",
        "metadata": {"tags": ["icd-colab-badge"]},
        "source": [
            f"[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})\n"
        ],
    }
    setup_source = f'''# Preparação automática para execução no Google Colab.
# Fora do Colab, esta célula não altera o diretório de trabalho.
try:
    import google.colab  # type: ignore
except ImportError:
    pass
else:
    import os
    import subprocess
    from pathlib import Path

    repository = Path("/content/icd")
    if not repository.exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/{REPOSITORY}.git", str(repository)
        ], check=True)
    os.chdir(repository / "{relative}")
    print("Material preparado em:", Path.cwd())
'''
    setup = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["icd-colab-setup"]},
        "outputs": [],
        "source": setup_source.splitlines(keepends=True),
    }
    notebook["cells"][:0] = [badge, setup]
    notebook.setdefault("metadata", {}).setdefault("colab", {})["name"] = (
        f"ICD — {directory.name}"
    )
    return notebook


def qmd_to_notebook(source: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="icd-colab-") as temporary:
        output_dir = Path(temporary)
        command = [
            "quarto",
            "render",
            str(source.relative_to(ROOT)),
            "--to",
            "ipynb",
            "--no-execute",
            "--output",
            "notebook-colab.ipynb",
            "--output-dir",
            str(output_dir),
            "--quiet",
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        with (output_dir / "notebook-colab.ipynb").open(encoding="utf-8") as stream:
            return json.load(stream)


def write_colab_notebook(source: Path) -> None:
    if source.suffix == ".qmd":
        notebook = qmd_to_notebook(source)
    else:
        with source.open(encoding="utf-8") as stream:
            notebook = json.load(stream)
    notebook = prepare_notebook(notebook, source.parent)
    target = source.parent / "notebook-colab.ipynb"
    with target.open("w", encoding="utf-8") as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
        stream.write("\n")


def add_button(class_page: Path, source: Path) -> None:
    text = class_page.read_text(encoding="utf-8")
    text = BUTTON_BLOCK.sub("\n", text)
    match = NOTEBOOK_LINK.search(text)
    if not match:
        raise RuntimeError(f"Notebook link not found in {class_page}")
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    block = (
        "\n<!-- colab-link:start -->\n"
        f"[Abrir no Google Colab]({colab_url(source.parent)})"
        "{.btn .btn-outline-primary target=\"_blank\"}\n"
        "<!-- colab-link:end -->"
    )
    text = text[:line_end] + block + text[line_end:]
    class_page.write_text(text, encoding="utf-8")


def main() -> None:
    pages = sorted((ROOT / "aulas").glob("[0-9][0-9]-*.qmd"))
    page_sources: list[tuple[Path, Path]] = []
    for page in pages:
        match = NOTEBOOK_LINK.search(page.read_text(encoding="utf-8"))
        if not match:
            raise RuntimeError(f"Notebook link not found in {page}")
        page_sources.append((page, ROOT / match.group("path")))

    sources = sorted({source for _, source in page_sources})
    for source in sources:
        write_colab_notebook(source)
    for page, source in page_sources:
        add_button(page, source)

    print(f"Generated {len(sources)} Colab notebooks for {len(pages)} class pages.")


if __name__ == "__main__":
    main()
