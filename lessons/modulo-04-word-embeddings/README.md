# Módulo 4, Word Embeddings

O Módulo 3 terminou com uma frustração produtiva. O Bag of Words e o TF-IDF
representam texto, mas não entendem nada, para eles cada palavra é uma ilha, sem
relação com as demais. Vimos isso doer quando uma pergunta de cálculo ficou mais
parecida com uma de programação por causa de palavras comuns. Este módulo resolve
esse problema de vez.

A ideia central é representar palavras e frases por vetores densos, posicionados de
modo que sentidos próximos fiquem perto no espaço. Vamos construir esses vetores por
três caminhos diferentes, prevendo contexto, olhando os pedaços das palavras e
fatorando coocorrências, e depois subir para o nível da frase, fechando com a busca
semântica que faltava desde o início. Cada aula implementa o seu método do zero, com
numpy, para você ver os embeddings emergirem de verdade.

## As quatro aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Word2Vec](01-word2vec.md) | Treina um skip-gram do zero e vê palavras próximas se aproximarem |
| 2 | [FastText](02-fasttext.md) | Representa palavras por n-gramas e trata palavras nunca vistas |
| 3 | [GloVe](03-glove.md) | Fatora a matriz de coocorrência com SVD para obter vetores |
| 4 | [Sentence Transformers](04-sentence-transformers.md) | Constrói a busca semântica de perguntas de alunos |

As três primeiras aulas tratam de vetores de palavra por abordagens
complementares, e a quarta dá o salto para vetores de frase, onde mora o projeto que
fecha o módulo.

## O que você leva deste módulo

Ao terminar, você entende por que os embeddings superam as representações esparsas,
conhece as três grandes famílias de embeddings de palavra e sabe construir uma busca
semântica que encontra textos pelo significado, e não pelas palavras exatas. Esse é o
alicerce direto do módulo de RAG, em que a busca vetorial é o coração do sistema.

## Projeto do módulo

O projeto integrador está na quarta aula, um buscador semântico de perguntas de
alunos que devolve as perguntas mais relacionadas a uma consulta, comparado com uma
busca tradicional por palavra-chave. O ponto alto é mostrar um caso em que a busca por
sentido acerta e a por palavra-chave falha, fechando o arco que começou no Bag of
Words.

## Pré-requisitos

O Módulo 3 e a noção de similaridade do cosseno são a base. Os notebooks usam numpy, e
a parte opcional da última aula usa a biblioteca sentence-transformers, que entra com
o ambiente de [docs/setup.md](../../docs/setup.md). Os exemplos principais rodam sem
ela, com a média de vetores construída do zero.

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-04/](../../notebooks/modulo-04/),
  um para cada aula.
- Glossário dos termos em [docs/glossario.md](../../docs/glossario.md).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
