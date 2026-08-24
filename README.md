# Introdução à Ciência de Dados

Repositório da disciplina de Introdução à Ciência de Dados.

## Estrutura

- `aulas/`: roteiros pedagógicos.
- `slides/`: slides em Quarto Reveal.js.
- `exemplos/`: exemplos documentados com dados, código e discussões.
- `listas/`: listas de exercícios.
- `vpl/`: templates e exercícios para Moodle/VPL.
- `.github/workflows/publish.yml`: publicação automática no GitHub Pages.

## Desenvolvimento local

```bash
quarto preview
```

## Publicação no GitHub

1. Crie um repositório vazio chamado `icd` no GitHub.
2. Atualize `site-url` e `repo-url` em `_quarto.yml`.
3. Envie o conteúdo para o repositório.
4. Em `Settings > Pages`, selecione `GitHub Actions` como fonte de publicação.

Depois do primeiro push na branch `main`, o site ficará disponível em:

```text
https://heitorramos.github.io/icd/
```

## Próxima etapa

Escolher uma aula piloto e substituir os templates por conteúdo real.
