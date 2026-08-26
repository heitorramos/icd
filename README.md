# Introdução à Ciência de Dados

Repositório da disciplina de Introdução à Ciência de Dados.

## Estrutura

## Regra de sincronização pedagógica

Slides e materiais de apoio formam uma única unidade didática. Sempre que um
slide receber uma definição, derivação, condição de validade, interpretação ou
limitação nova, o notebook correspondente deve ser atualizado no mesmo trabalho.
Os notebooks devem funcionar como guias autônomos de estudo: contextualizam a
base, reproduzem as análises, apresentam os resultados e desenvolvem com mais
detalhe a teoria introduzida nos slides.

- `aulas/`: roteiros pedagógicos.
- `slides/`: slides em Quarto Reveal.js.
- `exemplos/`: exemplos documentados com dados, código e discussões.
- `.github/workflows/publish.yml`: publicação automática no GitHub Pages.

## Desenvolvimento local

```bash
quarto preview
```

## Padrão para bases de dados nas aulas

Toda aula que utilizar uma base de dados deve apresentar, antes da análise
principal, uma descrição curta contendo:

- origem, contexto e objetivo da base;
- unidade de observação;
- principais colunas utilizadas e seus significados;
- tamanho, valores ausentes e categorias relevantes;
- uma ou duas visualizações ou estatísticas descritivas ligadas à pergunta da aula.

Esse panorama deve ocupar normalmente um ou dois slides. Toda figura deve incluir
um bloco recolhível **Código do gráfico** com o trecho que a produziu.

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
