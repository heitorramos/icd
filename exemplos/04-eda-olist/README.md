# Aula 04 — Análise Exploratória com a Olist

Material de apoio da Aula 04 de Introdução à Ciência de Dados.

## Arquivos

- `notebook.ipynb`: notebook executado para estudo.
- `notebook.qmd`: fonte editável do notebook.
- `data/`: amostra reprodutível de 20 mil pedidos e linhas relacionadas.
- `generate_figures.py`: gera as figuras usadas nos slides.

## Base

Fonte original: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

O conjunto original possui aproximadamente 100 mil pedidos entre 2016 e 2018. A amostra didática foi obtida com `orders.sample(n=20000, random_state=42)`; todas as linhas relacionadas das tabelas de clientes, itens, pagamentos e avaliações foram preservadas.

## Unidade de observação

Cada arquivo possui uma granularidade distinta. O notebook ensina a agregar as tabelas à unidade de pedido antes de construir uma tabela analítica.
