# Módulo 6, Transformers

O Módulo 5 terminou mostrando os limites das redes recorrentes, a memória que some em
sequências longas e a lentidão de processar uma palavra por vez. Este módulo apresenta a
arquitetura que resolveu os dois problemas e passou a dominar todo o NLP, o Transformer.
A sua ideia central, a atenção, deixa cada palavra olhar para todas as outras de uma vez,
em paralelo, não importa a distância.

A trilha do módulo monta o Transformer peça por peça. Começamos pela self-attention,
depois a multiplicamos em várias cabeças, então a envolvemos no bloco completo com encoder
e decoder. Com o bloco pronto, conhecemos os dois grandes modelos que nascem dele, o BERT,
que lê nos dois sentidos para entender, e o GPT, que gera texto olhando só para trás. Cada
aula implementa os mecanismos do zero, com numpy, para você ver a atenção funcionando de
verdade.

## As cinco aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Self-attention](01-self-attention.md) | Implementa a atenção e lê a matriz de atenção |
| 2 | [Multi-head attention](02-multi-head-attention.md) | Roda várias cabeças e compara o que cada uma vê |
| 3 | [Encoder e decoder](03-encoder-decoder.md) | Monta um bloco completo com posição, residual e norma |
| 4 | [BERT](04-bert.md) | Demonstra a força do contexto bidirecional |
| 5 | [GPT](05-gpt.md) | Implementa a atenção causal e gera texto |

As três primeiras aulas constroem a maquinaria da atenção, e as duas últimas mostram as
duas grandes aplicações dela, entender com o BERT e gerar com o GPT.

## O que você leva deste módulo

Ao terminar, você entende por dentro a atenção, sabe distinguir um modelo de encoder de um
de decoder, e consegue explicar por que os Transformers substituíram as RNN. Mais
importante, você passa a enxergar os LLMs não como mágica, mas como pilhas de blocos de
atenção treinados para prever palavras, o que prepara diretamente o Módulo 7, dedicado aos
grandes modelos de linguagem.

## Projeto do módulo

O projeto integrador está na quinta aula, implementar a self-attention e a atenção causal do
zero, verificar as suas propriedades, e comparar o resultado com uma referência de
biblioteca ou ver a arquitetura em ação gerando texto com um modelo de verdade. É a ponte
concreta entre a teoria da atenção e os modelos que conversam com a gente.

## Pré-requisitos

Os Módulos 4 e 5 ajudam, pois o módulo combina embeddings com redes neurais. Os notebooks
usam numpy e rodam sem instalar mais nada. As partes opcionais das aulas de BERT e GPT usam
a biblioteca transformers e o Ollama, que entram com o ambiente de
[docs/setup.md](../../docs/setup.md).

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-06/](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-06),
  um para cada aula.
- Glossário dos termos em [docs/glossario.md](../../docs/glossario.md).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
