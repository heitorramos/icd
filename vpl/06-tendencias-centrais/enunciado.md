# VPL 06: tendências centrais

## Objetivo

Implementar funções básicas de análise descritiva para uma coluna de durações de
músicas.

Este VPL acompanha a Aula 06. Ele usa uma versão pequena e autocontida do
problema trabalhado no notebook: converter durações no formato `minutos:segundos`
e calcular média, mediana, quartis e um resumo descritivo.

## Arquivo a editar

Edite apenas o arquivo `template.py`.

## Funções obrigatórias

### `duration_to_minutes(duration)`

Recebe uma string no formato `"M:SS"` e retorna a duração em minutos como número
decimal.

Exemplo:

```python
duration_to_minutes("3:30") == 3.5
```

### `media(dados)`

Recebe uma sequência de números e retorna a média aritmética.

### `mediana(dados)`

Recebe uma sequência de números e retorna a mediana.

Para quantidade par de valores, retorne a média dos dois valores centrais.

### `quantil(dados, p)`

Recebe uma sequência de números e um valor `p` entre 0 e 1. Retorna o quantil
usando interpolação linear entre posições ordenadas:

```text
posição = p * (n - 1)
```

Se a posição não for inteira, interpole entre os dois vizinhos.

### `quartis(dados)`

Retorna uma tupla `(q1, q2, q3)`, correspondente aos quantis 0,25, 0,50 e 0,75.

### `resumo_duracoes(duracoes)`

Recebe uma sequência de strings no formato `"M:SS"` e retorna um dicionário com:

```python
{
    "n": quantidade_de_observacoes,
    "media": media_em_minutos,
    "mediana": mediana_em_minutos,
    "q1": primeiro_quartil,
    "q3": terceiro_quartil,
    "intervalo": maximo_menos_minimo,
}
```

## Critérios de avaliação

- Os testes públicos verificam casos básicos.
- Os testes ocultos verificam listas em outra ordem, quantidade par de valores e
  durações diferentes.
- A solução não deve usar valores fixos dependentes dos exemplos.
- A solução pode usar apenas Python padrão; não é necessário usar `pandas` ou
  `numpy`.
