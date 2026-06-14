# Módulo 8, Prompt Engineering

O Módulo 7 nos mostrou o que é um LLM. Este módulo ensina a conversar com ele de forma
eficaz, que é o prompt engineering. Um modelo ajustado por instrução já sabe seguir pedidos,
mas a qualidade da resposta depende muito de como o pedido é feito. Aprender a formular bons
prompts é a habilidade que mais rende resultado imediato ao trabalhar com LLMs, e não exige
treinar nada.

A trilha vai do simples ao estruturado. Começamos pedindo tarefas sem exemplos, com o
zero-shot, depois acrescentamos exemplos com o few-shot, em seguida fazemos o modelo raciocinar
com o Chain of Thought, e por fim o obrigamos a responder em formato fixo com a saída
estruturada. Cada técnica resolve um tipo de problema, e juntas formam a caixa de ferramentas
do prompt engineering. As partes determinísticas, como montar prompts e validar JSON, rodam em
Python puro, e as respostas de verdade vêm de um LLM local via Ollama.

## As quatro aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Zero-shot](01-zero-shot.md) | Compara prompts vagos com prompts bem estruturados |
| 2 | [Few-shot](02-few-shot.md) | Guia o modelo com exemplos no próprio prompt |
| 3 | [Chain of Thought](03-chain-of-thought.md) | Faz o modelo raciocinar passo a passo |
| 4 | [Structured output](04-structured-output.md) | Obtém e valida saída em JSON |

As três primeiras aulas tratam de como pedir e guiar o modelo, e a quarta de como fazer a
saída ser usável por outros programas, onde mora o projeto integrador.

## O que você leva deste módulo

Ao terminar, você sabe formular prompts claros, guiar o modelo com exemplos, ativar o raciocínio
quando preciso, e obter saídas estruturadas e confiáveis. Essa última habilidade é especialmente
importante, porque é o que permite ligar um LLM ao código, e é a base direta dos agentes, que
veremos no Módulo 10.

## Projeto do módulo

O projeto integrador está na quarta aula, um tutor que explica um conceito em três níveis de
profundidade e devolve cada explicação em formato estruturado, combinando todas as técnicas do
módulo, prompt claro, exemplos, raciocínio quando útil e saída em JSON validada.

## Pré-requisitos

O Módulo 7 sobre LLMs é a base. As partes determinísticas dos notebooks rodam só com a
biblioteca padrão do Python. As demonstrações com respostas de verdade usam o Ollama, conforme
o [docs/setup.md](../../docs/setup.md), e degradam de forma graciosa quando ele não está
disponível.

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-08/](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-08),
  um para cada aula.
- Glossário dos termos em [docs/glossario.md](../../docs/glossario.md).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
