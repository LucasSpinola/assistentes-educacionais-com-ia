# Módulo 7, Large Language Models

O Módulo 6 nos deu a arquitetura Transformer. Este módulo mostra como, a partir dela, se
constrói um Large Language Model, e o que cada etapa de treino acrescenta. A mensagem
central é desmistificadora, um LLM não é mágica, é um Transformer decoder gigante treinado
para prever a próxima palavra, e depois refinado para seguir instruções e agradar as
pessoas.

A trilha segue a própria vida de um LLM. Primeiro entendemos a máquina e como ela gera
texto. Depois vemos o pré-treino, onde ela aprende a língua prevendo palavras em um corpus
imenso. Em seguida, o fine-tuning e o ajuste por instrução, que a transformam de um modelo
que completa texto em um assistente que responde. Por fim, o RLHF, que a alinha ao que as
pessoas preferem. Cada aula traz uma demonstração executável do conceito, com numpy ou com
um modelo local via Ollama.

## As quatro aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Como LLMs funcionam](01-como-llms-funcionam.md) | Demonstra a amostragem por temperatura e gera texto |
| 2 | [Pré-treino](02-pre-treino.md) | Mede a perplexidade, a métrica do pré-treino |
| 3 | [Fine-tuning e instrução](03-fine-tuning-instrucao.md) | Compara um modelo base com um de instrução |
| 4 | [RLHF](04-rlhf.md) | Constrói um modelo de recompensa a partir de preferências |

As duas primeiras aulas explicam o que é um LLM e como ele nasce, e as duas últimas mostram
como ele é refinado para virar um assistente útil e alinhado.

## O que você leva deste módulo

Ao terminar, você entende as etapas que transformam um Transformer em um assistente, sabe o
que é perplexidade, temperatura, ajuste por instrução e RLHF, e perde o medo da caixa-preta,
passando a enxergar os LLMs como sistemas compreensíveis, com forças e limites claros. Essa
clareza é o que vai te permitir usá-los bem nos módulos de prompt engineering, RAG e agentes.

## Projeto do módulo

O projeto integrador está na terceira aula, comparar as respostas de um modelo base e de um
modelo ajustado por instrução para um conjunto de perguntas educacionais, registrando o que
o ajuste por instrução muda no comportamento. É a forma mais direta de sentir o efeito de
uma das etapas mais importantes do treino de um LLM.

## Pré-requisitos

O Módulo 6 sobre Transformers é a base direta. Os exemplos numéricos rodam com numpy, e as
demonstrações com modelos de verdade usam o Ollama, conforme o
[docs/setup.md](../../docs/setup.md). Tudo o que depende de modelo externo degrada de forma
graciosa quando ele não está disponível.

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-07/](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-07),
  um para cada aula.
- Glossário dos termos em [docs/glossario.md](../../docs/glossario.md).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
