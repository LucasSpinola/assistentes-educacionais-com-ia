# Módulo 10, Agentes

Até o Módulo 9, usamos o LLM como alguém que responde. Este módulo dá a ele a capacidade de
agir. Um agente é um LLM colocado dentro de um ciclo, capaz de usar ferramentas, observar
resultados e continuar, até concluir uma tarefa. É o que transforma um modelo que conversa em
um sistema que resolve problemas de várias etapas, como calcular, buscar no material e
encadear passos.

A trilha constrói o agente peça por peça. Começamos pelo agent loop, o ciclo de perceber,
decidir, agir e observar. Depois damos ao LLM o comando, com o tool calling. Em seguida, o
planejamento de várias etapas, a memória sobre o aluno, e a orquestração em grafo com
LangGraph. Cada aula implementa a peça do zero, com Python puro, e mostra a ferramenta de
verdade como opção. No fim, tudo se junta no projeto, um agente tutor que usa ferramentas e
lembra do aluno.

## As cinco aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Agent loop](01-agente-e-agent-loop.md) | Constrói um agente que escolhe entre ferramentas |
| 2 | [Tool calling](02-tool-calling.md) | Faz o LLM pedir ferramentas em JSON e despacha |
| 3 | [Planning](03-planning.md) | Resolve problemas de várias etapas encadeadas |
| 4 | [Memory](04-memory.md) | Dá memória de curto e de longo prazo ao agente |
| 5 | [Orquestração com LangGraph](05-orquestracao-langgraph.md) | Modela o agente como grafo de estados |

As quatro primeiras aulas constroem as capacidades do agente, e a quinta as orquestra e abriga
o projeto do agente tutor.

## O que você leva deste módulo

Ao terminar, você sabe construir um agente que usa ferramentas, planeja em etapas e mantém
memória, entende o agent loop e o tool calling por dentro, e tem um agente tutor funcional no
portfólio. Esse conhecimento é a base direta do Módulo 11, em que vários agentes passam a
cooperar, e do projeto final da trilha.

## Projeto do módulo

O projeto integrador está na pasta
[projects/m10-tutor-agent/](../../projects/m10-tutor-agent/), um agente tutor que usa uma
calculadora segura e uma busca no material, decide qual ferramenta usar, planeja problemas de
várias etapas e lembra de informações do aluno para personalizar. Ele roda do zero, sem
dependências, e tem testes.

## Pré-requisitos

Os Módulos 7 (LLMs), 8 (prompting e saída estruturada) e 9 (RAG) são a base direta. Os
exemplos principais e o projeto rodam só com a biblioteca padrão do Python. As versões com o
LLM como controlador e com o LangGraph usam o Ollama e bibliotecas, conforme o
[docs/setup.md](../../docs/setup.md), e degradam de forma graciosa quando ausentes.

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-10/](../../notebooks/modulo-10/),
  um para cada aula.
- Projeto completo em [projects/m10-tutor-agent/](../../projects/m10-tutor-agent/).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
