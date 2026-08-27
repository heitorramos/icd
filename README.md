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

As páginas e os materiais de apoio não devem incluir seções de
**Dimensionamento para 100 minutos**. A organização do tempo de aula não faz
parte do material destinado aos estudantes.

Discussões de resultados devem ser específicas para a saída imediatamente
anterior. Evite parágrafos genéricos repetidos entre células ou seções; mantenha,
em geral, uma interpretação por etapa analítica, mencionando o padrão observado,
a unidade de análise e a limitação relevante.

Toda equação apresentada em slides, páginas ou materiais de apoio deve definir
imediatamente seus termos: símbolos, índices, parâmetros, estatísticas, conjuntos
e operadores menos usuais. Também deve indicar as condições ou hipóteses
necessárias e traduzir o resultado para linguagem natural. Quando a equação for
aplicada a um exemplo, explicite a correspondência entre a notação e as variáveis
da base utilizada. Não presuma que a notação seja autoexplicativa.

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

Na página da aula, toda base utilizada deve aparecer na grade inicial em um card
intitulado **Base de dados de apoio**, com uma descrição breve e link para a
fonte. Quando houver mais de uma base, o mesmo card pode reunir os links, deixando
claro qual é a base principal e qual sustenta exemplos complementares.

## Padrão para bibliografia por aula

Toda página de aula deve terminar com a seção **Bibliografia da aula**, relacionando
capítulos ou seções específicos aos conteúdos ensinados. A seção deve distinguir:

- **Bibliografia principal**, priorizando os livros adotados na disciplina;
- **Bibliografia complementar**, quando o conteúdo não estiver suficientemente
  coberto nos textos principais.

As referências devem incluir links sempre que houver uma versão oficial ou aberta
e indicar brevemente quais tópicos da aula cada leitura sustenta. Esse padrão deve
ser aplicado também às próximas aulas.

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
