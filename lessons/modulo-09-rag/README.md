# Módulo 9, Retrieval-Augmented Generation

Um LLM sozinho tem dois problemas para um assistente educacional: ele só sabe o que viu no
treino, e às vezes inventa. O RAG resolve os dois dando ao modelo uma memória externa de
documentos. Antes de responder, o sistema busca os trechos relevantes do material e os
entrega ao modelo, que então responde com base neles, com menos alucinação e com
conhecimento atualizado e específico. Este é o primeiro módulo com um projeto de assistente
completo.

A trilha monta o assistente peça por peça. Começamos com o pipeline de RAG mínimo, depois
melhoramos os embeddings e o chunking, em seguida guardamos os vetores em um banco vetorial,
montamos a busca e o contexto com citação das fontes, e por fim olhamos para a produção com o
Qdrant. Cada aula constrói uma peça do zero, em Python puro, e mostra a ferramenta de verdade
como caminho opcional. No fim, tudo se junta no projeto, um assistente educacional que
responde citando o material.

## As cinco aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Pipeline de RAG](01-pipeline-de-rag.md) | Monta um RAG mínimo de ponta a ponta |
| 2 | [Embeddings e vetores](02-embeddings-e-vetores.md) | Faz chunking e compara embeddings |
| 3 | [ChromaDB na prática](03-chromadb-na-pratica.md) | Constrói um banco vetorial do zero |
| 4 | [Busca semântica e contexto](04-busca-semantica-contexto.md) | Monta o contexto com citação e trata a ausência |
| 5 | [Qdrant para produção](05-qdrant-producao.md) | Filtra por metadados e escala (projeto do módulo) |

As quatro primeiras aulas constroem as peças do RAG, e a quinta olha para a produção e abriga
o assistente completo.

## O que você leva deste módulo

Ao terminar, você sabe construir um sistema de RAG de ponta a ponta, entende o papel dos
embeddings, do banco vetorial, da montagem de contexto e do tratamento da ausência, e tem um
assistente educacional funcional no portfólio. Esse conhecimento é a base do Módulo 14, e a
busca de conhecimento é uma das ferramentas centrais dos agentes do Módulo 10.

## Projeto do módulo

O projeto integrador está na pasta
[projects/m09-rag-assistant/](../../projects/m09-rag-assistant/), um assistente educacional
que indexa notas de aula com metadados, recupera trechos relevantes, responde citando as
fontes, recusa de forma honesta o que não está no material e filtra a busca por disciplina.
Ele roda do zero, sem dependências, e tem testes.

## Pré-requisitos

Os Módulos 4 (embeddings), 7 (LLMs) e 8 (prompting) são a base direta. Os exemplos
principais e o projeto rodam só com a biblioteca padrão do Python. As versões com embeddings
densos, ChromaDB, Qdrant e geração com LLM usam bibliotecas e o Ollama, conforme o
[docs/setup.md](../../docs/setup.md), e degradam de forma graciosa quando ausentes.

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-09/](../../notebooks/modulo-09/),
  um para cada aula.
- Projeto completo em [projects/m09-rag-assistant/](../../projects/m09-rag-assistant/).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
