# Módulo 5, Deep Learning para NLP

Os módulos anteriores nos deram representações de texto cada vez melhores, mas os
modelos que usamos para classificar ainda eram simples. Este módulo traz o motor que
move a IA moderna, as redes neurais profundas, e o adapta para o que o texto tem de
especial, ser uma sequência que se desenrola no tempo.

A trilha do módulo é uma história de problemas e soluções. Começamos construindo uma
rede neural do zero e resolvendo o XOR que derrubou o perceptron lá no Módulo 1. Depois
damos memória à rede com as RNN, e descobrimos a sua fraqueza com sequências longas.
Então conhecemos a LSTM e a GRU, que curam essa fraqueza com portões. Cada aula
implementa as ideias na unha, com numpy, e a última também mostra a versão de produção
com PyTorch.

## As quatro aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Redes neurais](01-redes-neurais.md) | Constrói uma rede com backpropagation e resolve o XOR |
| 2 | [RNN](02-rnn.md) | Treina uma RNN e vê o gradiente sumir em sequências longas |
| 3 | [LSTM](03-lstm.md) | Demonstra a esteira de memória preservando informação |
| 4 | [GRU](04-gru.md) | Implementa a GRU e treina um classificador de intenção |

A primeira aula é a fundação, as duas seguintes mostram o problema da memória e a sua
solução, e a quarta fecha com o projeto integrador, onde mora o classificador de
intenção de mensagens de alunos.

## O que você leva deste módulo

Ao terminar, você entende por dentro a backpropagation, sabe por que sequências exigem
arquiteturas próprias, e conhece as três grandes células recorrentes e as suas trocas.
Também leva a percepção de que a recorrência, apesar de poderosa, tem limites de memória
e de velocidade, o que prepara o terreno para os Transformers do Módulo 6, que trocam a
recorrência pela atenção.

## Projeto do módulo

O projeto integrador está na quarta aula, um classificador neural de intenção de
mensagens curtas de alunos, que separa dúvidas, elogios e problemas técnicos. A versão
principal roda do zero, com Bag of Words e a rede neural da primeira aula, e há uma
versão avançada opcional com uma GRU em PyTorch, que leva em conta a ordem das palavras.

## Pré-requisitos

Os Módulos 2, 3 e 4 ajudam bastante, pois o módulo combina gradiente descendente,
representação de texto e embeddings. Os notebooks usam numpy e rodam sem instalar mais
nada. A parte opcional da última aula usa PyTorch, que entra com o ambiente de
[docs/setup.md](../../docs/setup.md).

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-05/](../../notebooks/modulo-05/),
  um para cada aula.
- Convenções de notação em [docs/notacao-matematica.md](../../docs/notacao-matematica.md).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
