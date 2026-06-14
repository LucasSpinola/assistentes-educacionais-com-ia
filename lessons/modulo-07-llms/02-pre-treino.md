# Aula 2, Pré-treino

> Esta aula explica de onde vem o conhecimento de um LLM, o pré-treino. Antes de
> qualquer ajuste, o modelo passa meses prevendo a próxima palavra em um corpus
> gigantesco, e é aí que ele aprende a língua e o mundo. Vamos medir a perplexidade,
> a métrica que avalia essa previsão.

Na aula anterior, vimos que um LLM gera texto prevendo a próxima palavra. Mas como ele
aprende a prever bem? A resposta é o pré-treino, a fase mais longa e mais cara da vida de
um LLM. Nela, o modelo é exposto a uma quantidade imensa de texto e treinado, sem nenhum
rótulo humano, apenas para acertar a próxima palavra de cada trecho.

Esse treino é chamado de auto-supervisionado, porque o próprio texto fornece a resposta
correta, a palavra que de fato vinha a seguir. Repetido trilhões de vezes, esse exercício
simples força o modelo a internalizar gramática, fatos, estilos e até algum raciocínio,
pois tudo isso ajuda a prever melhor. Nesta aula você vai entender o objetivo do pré-treino,
conhecer as leis de escala que guiam o tamanho dos modelos, e medir a perplexidade, a
métrica que diz o quão bem um modelo prevê texto.

---

## Objetivos

Ao final desta aula, você deve ser capaz de:

- Explicar o objetivo de pré-treino por previsão da próxima palavra.
- Entender por que ele é auto-supervisionado e dispensa rótulos.
- Compreender a perplexidade como medida de qualidade da previsão.
- Conhecer a ideia das leis de escala entre dados, parâmetros e computação.

## Teoria

O pré-treino otimiza um objetivo simples, a entropia cruzada da previsão da próxima
palavra, somada sobre todo o corpus. Para cada posição, o modelo produz uma distribuição
sobre o vocabulário e é penalizado conforme erra a palavra verdadeira. Como a resposta
certa é simplesmente a próxima palavra do texto, não é preciso ninguém rotular nada, o que
permite usar corpora enormes, de bilhões a trilhões de palavras.

A qualidade dessa previsão é medida pela perplexidade, que pode ser entendida como o número
médio de palavras entre as quais o modelo hesita a cada passo. Um modelo que sempre acerta
teria perplexidade 1, e um que chuta ao acaso teria perplexidade igual ao tamanho do
vocabulário. Quanto menor a perplexidade, melhor o modelo previu o texto.

```mermaid
flowchart LR
    C[Corpus gigante sem rótulos] --> O[Objetivo: prever a próxima palavra]
    O --> T[Treino auto-supervisionado]
    T --> M[Modelo base, com conhecimento de língua e mundo]
```

O quanto investir em cada ingrediente é tema das leis de escala. Kaplan e colegas mostraram
que o desempenho melhora de forma previsível conforme crescem os dados, os parâmetros e a
computação. Mais tarde, o trabalho do Chinchilla, de Hoffmann e colegas, refinou essa
receita, mostrando que muitos modelos eram grandes demais para a quantidade de dados com
que foram treinados, e que equilibrar melhor os dois rende modelos melhores com o mesmo
custo.

## Explicação Intuitiva

Imagine alguém que leu praticamente tudo, e cujo único exercício a vida toda foi adivinhar a
próxima palavra de cada texto que encontrou. Para ficar bom nisso, essa pessoa precisou,
sem perceber, aprender gramática, porque ajuda a prever a concordância, aprender fatos,
porque a próxima palavra muitas vezes depende deles, e até aprender a seguir um raciocínio,
porque textos lógicos têm continuações previsíveis. O pré-treino é esse exercício, levado a
uma escala sobre-humana.

A perplexidade é como medir o tamanho da dúvida dessa pessoa a cada palavra. Se a cada
passo ela hesita entre poucas opções, está prevendo bem, e a perplexidade é baixa. Se hesita
entre muitas, está perdida, e a perplexidade é alta. Acompanhar a perplexidade cair ao longo
do treino é ver o modelo aprendendo a língua diante dos nossos olhos.

## Explicação Matemática

O objetivo de pré-treino é minimizar a entropia cruzada da previsão da próxima palavra.
Para um corpus com tokens $w_1, \dots, w_N$, a perda é

$$
\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log P(w_t \mid w_1, \dots, w_{t-1}).
$$

A perplexidade é a exponencial dessa perda média,

$$
\text{PPL} = \exp(\mathcal{L}).
$$

A interpretação é elegante. Se o modelo atribui, em média, probabilidade $p$ à palavra
correta, a perplexidade fica em torno de $1/p$, ou seja, o número efetivo de opções entre as
quais ele hesita. Um modelo aleatório sobre um vocabulário de tamanho $V$ tem perplexidade
$V$. Qualquer perplexidade bem abaixo de $V$ significa que o modelo aprendeu estrutura real
da língua.

## Exemplo Prático

Para tornar a perplexidade concreta, vamos construir um modelo de bigramas bem simples sobre
um pequeno corpus, no espírito do gerador do Módulo 1, e calcular a sua perplexidade. Não é
um LLM, claro, mas a métrica é exatamente a mesma usada para avaliar o pré-treino dos
grandes modelos.

A expectativa, que vamos confirmar, é uma perplexidade bem menor que o tamanho do
vocabulário, sinal de que mesmo um modelo tão simples já capturou parte da estrutura do
texto. O código está no notebook
[notebooks/modulo-07/02-pre-treino.ipynb](../../notebooks/modulo-07/02-pre-treino.ipynb),
então abra-o ao lado para acompanhar.

## Código Comentado

```python
import math
from collections import Counter, defaultdict

corpus = ("o aluno estuda matematica todos os dias o aluno resolve exercicios "
          "o aluno aprende matematica a professora explica a materia").split()

# Modelo de bigramas: para cada palavra, conta o que vem depois.
vocab = set(corpus)
modelo = defaultdict(Counter)
for a, b in zip(corpus, corpus[1:]):
    modelo[a][b] += 1


def perplexidade(tokens):
    """Perplexidade do modelo de bigramas, com suavização de Laplace."""
    soma_log = 0.0
    n = 0
    for a, b in zip(tokens, tokens[1:]):
        cont = modelo.get(a, Counter())
        total = sum(cont.values())
        prob = (cont[b] + 1) / (total + len(vocab))   # Laplace
        soma_log += math.log(prob)
        n += 1
    return math.exp(-soma_log / n)


ppl = perplexidade(corpus)
print(f"Vocabulário: {len(vocab)} palavras")
print(f"Perplexidade no corpus: {ppl:.2f}")
print(f"Perplexidade de um modelo aleatório seria perto de {len(vocab)}")
```

Ao rodar, a perplexidade fica em torno de 7, enquanto um modelo que chutasse ao acaso teria
perplexidade perto de 14, o tamanho do vocabulário. Mesmo um modelo de bigramas, que só olha
a palavra anterior, já reduz pela metade a dúvida a cada passo, porque aprendeu quais
palavras costumam se seguir. Um LLM faz isso de forma incomparavelmente melhor, considerando
todo o contexto, e por isso atinge perplexidades muito mais baixas, que é o sinal de que ele
domina a língua.

## Exercícios

1) Conceitual: Por que o pré-treino é chamado de auto-supervisionado? De onde vem a resposta
   correta?
2) Conceitual: O que significa uma perplexidade de 10, em termos de hesitação do modelo a
   cada palavra?
3) Prático: Aumente o corpus com mais frases e veja se a perplexidade do modelo de bigramas
   muda.
4) Prático: Implemente um modelo de trigramas e compare a sua perplexidade com a do bigrama.
5) Extensão: Pesquise as leis de escala do Chinchilla e resuma, em um parágrafo, o que elas
   recomendam sobre equilibrar dados e parâmetros.

## Projeto da Aula

Compare a perplexidade de modelos de ordens diferentes. A entrega é um experimento que treina
modelos de unigrama, bigrama e trigrama sobre o mesmo corpus e compara as suas perplexidades,
mostrando como olhar mais contexto reduz a dúvida do modelo.

Considere o projeto pronto quando você tiver as três perplexidades e um parágrafo explicando
por que mais contexto costuma ajudar, e também por que modelos de ordem muito alta podem
sofrer com dados escassos, retomando a ideia de overfitting. Esse entendimento da métrica de
pré-treino prepara a próxima aula, em que pegamos um modelo já pré-treinado e o ajustamos
para tarefas específicas.

## Leituras Recomendadas

- O artigo das leis de escala, de Kaplan e colegas, sobre como o desempenho cresce com a
  escala.
- O artigo do Chinchilla, de Hoffmann e colegas, sobre treino com computação ótima.
- Capítulos sobre modelos de linguagem e perplexidade em Jurafsky e Martin, Speech and
  Language Processing.

## Referências Científicas

As referências abaixo são reais e estão registradas em
[references/referencias.bib](../../references/referencias.bib). As chaves entre
parênteses são as do BibTeX.

- Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. (`kaplan2020scaling`)
- Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. NeurIPS.
  (`hoffmann2022chinchilla`)
- Jurafsky, D., e Martin, J. H. (2009). Speech and Language Processing, 2ª edição. Pearson
  Prentice Hall. (`jurafsky2009slp`)
