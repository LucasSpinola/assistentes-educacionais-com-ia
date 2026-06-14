# Módulo 2, Fundamentos de Machine Learning

Se o Módulo 1 deu o mapa das ideias, este é onde a gente começa a construir de
verdade. Os fundamentos de Machine Learning são a base estatística sobre a qual se
apoiam praticamente todos os módulos seguintes, dos embeddings aos grandes modelos
de linguagem. Em vez de programar o comportamento, aqui aprendemos a programar a
capacidade de aprender a partir de dados.

Para que nada fique abstrato, um mesmo cenário acompanha as quatro aulas, o de
prever e avaliar o desempenho de alunos. Começamos estimando uma nota, depois
prevemos uma aprovação, em seguida enfrentamos a armadilha de um modelo que decora
em vez de aprender, e por fim aprendemos a medir desempenho com honestidade. Cada
aula constrói o seu método do zero, em Python e numpy, para que você veja a engrenagem
por dentro antes de usar bibliotecas prontas.

## As quatro aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Regressão](01-regressao.md) | Implementa uma regressão linear com gradiente descendente |
| 2 | [Classificação](02-classificacao.md) | Constrói uma regressão logística e mede a acurácia |
| 3 | [Overfitting e underfitting](03-overfitting-underfitting.md) | Faz o erro de teste desenhar o U do overfitting |
| 4 | [Validação](04-validacao.md) | Avalia um modelo com validação cruzada feita à mão |

As aulas formam uma progressão. As duas primeiras apresentam os dois grandes tipos
de tarefa supervisionada. A terceira mostra o que dá errado quando ignoramos a
generalização. A quarta dá as ferramentas para medir desempenho sem se enganar, e é
onde mora o projeto que fecha o módulo.

## O que você leva deste módulo

Ao terminar, você entende o ciclo completo do aprendizado supervisionado, partir de
dados, ajustar um modelo, diagnosticar problemas de generalização e avaliar o
resultado de forma confiável. Você também ganha intuição sobre gradiente descendente,
funções de custo e o equilíbrio entre viés e variância, conceitos que vão reaparecer,
em maior escala, quando chegarmos às redes neurais e aos Transformers.

## Projeto do módulo

O projeto integrador está na quarta aula. A proposta é treinar e avaliar um
classificador de aprovação de alunos, comparando de forma honesta o desempenho no
treino com o desempenho estimado por validação cruzada, e interpretar a diferença à
luz do que se aprendeu sobre overfitting. É um exercício pequeno, mas que percorre
todo o caminho, do modelo à avaliação.

## Pré-requisitos

O Módulo 1 e noções básicas de álgebra e estatística ajudam bastante. Para rodar os
notebooks, basta Python e o ambiente de [docs/setup.md](../../docs/setup.md). O
núcleo dos exemplos usa apenas numpy, e as partes com gráficos ou scikit-learn são
opcionais e degradam de forma graciosa quando a biblioteca não está instalada.

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-02/](../../notebooks/modulo-02/),
  um para cada aula.
- Convenções de notação matemática em
  [docs/notacao-matematica.md](../../docs/notacao-matematica.md).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
