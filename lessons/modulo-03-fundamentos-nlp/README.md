# Módulo 3, Fundamentos de NLP

Um assistente educacional vive de texto, as perguntas dos alunos chegam em
linguagem natural, e é preciso entendê-las. Mas modelos trabalham com números, não
com frases. Este módulo é sobre essa travessia, como transformar texto em algo que
um computador consegue processar, usando as técnicas clássicas de Processamento de
Linguagem Natural que ainda hoje formam a base do campo.

Para não ficar no abstrato, uma mesma coleção de perguntas de alunos atravessa as
cinco aulas e vai sendo lapidada a cada etapa. Primeiro quebramos o texto em tokens,
depois removemos as palavras vazias, em seguida aproximamos variações de uma mesma
palavra, então viramos tudo em vetores de contagem e, por fim, damos pesos
inteligentes a cada palavra. No final, esses passos se juntam em um classificador que
descobre o tema de uma pergunta, que é o projeto que fecha o módulo.

## As cinco aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Tokenização](01-tokenizacao.md) | Constrói um tokenizador que trata pontuação e maiúsculas |
| 2 | [Stopwords](02-stopwords.md) | Remove palavras vazias e vê a lei de Zipf nos dados |
| 3 | [Stemming e lemmatização](03-stemming-lemmatizacao.md) | Implementa um stemmer simples para o português |
| 4 | [Bag of Words](04-bag-of-words.md) | Vira texto em vetores e mede similaridade do cosseno |
| 5 | [TF-IDF](05-tf-idf.md) | Dá pesos às palavras e classifica perguntas por tema |

As aulas se encadeiam, e o texto processado em uma vira a entrada da seguinte. As
três primeiras tratam do pré-processamento, e as duas últimas, da representação
vetorial, onde mora o projeto integrador.

## O que você leva deste módulo

Ao terminar, você sabe pegar uma frase crua e prepará-la para qualquer tarefa de
texto, entende por que algumas palavras valem mais que outras, e consegue construir
um classificador de texto simples do zero. Também leva uma percepção importante, a de
que essas representações clássicas ignoram a ordem e o significado das palavras, uma
limitação que abre caminho para os embeddings do Módulo 4.

## Projeto do módulo

O projeto integrador está na quinta aula, um classificador de perguntas de alunos por
tema, construído de ponta a ponta com tudo o que o módulo ensinou, tokenização,
remoção de stopwords, normalização, vetores TF-IDF e classificação por similaridade
do cosseno. É um sistema pequeno, transparente e que de fato funciona, uma boa peça
de portfólio.

## Pré-requisitos

O Módulo 2 ajuda a entender a parte de classificação e avaliação. Os notebooks deste
módulo são em Python puro, com as bibliotecas `re`, `collections` e `math`, então
rodam sem instalar nada além do Python descrito em
[docs/setup.md](../../docs/setup.md). As bibliotecas dedicadas de NLP, como NLTK e
spaCy, aparecem como sugestão de aprofundamento.

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-03/](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-03),
  um para cada aula.
- Glossário dos termos de NLP em [docs/glossario.md](../../docs/glossario.md).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
