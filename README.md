# Introducao a Ciencia de Dados

Repositorio da disciplina de Introducao a Ciencia de Dados.

## Estrutura

- `aulas/`: roteiros pedagogicos.
- `slides/`: slides em Quarto Reveal.js.
- `exemplos/`: exemplos documentados com dados, codigo e discussoes.
- `listas/`: listas de exercicios.
- `vpl/`: templates e exercicios para Moodle/VPL.
- `.github/workflows/publish.yml`: publicacao automatica no GitHub Pages.

## Desenvolvimento local

```bash
quarto preview
```

## Publicacao no GitHub

1. Crie um repositorio vazio chamado `icd` no GitHub.
2. Atualize `site-url` e `repo-url` em `_quarto.yml`.
3. Envie o conteudo para o repositorio.
4. Em `Settings > Pages`, selecione `GitHub Actions` como fonte de publicacao.

Depois do primeiro push na branch `main`, o site ficara disponivel em:

```text
https://heitorramos.github.io/icd/
```

## Proxima etapa

Escolher uma aula piloto e substituir os templates por conteudo real.
