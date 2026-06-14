# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Referência: Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., e Kiela, D. (2020). Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems (NeurIPS).
arXiv:2005.11401, https://arxiv.org/abs/2005.11401. Chave no BibTeX: `lewis2020rag`.

## Problema que ataca

Modelos de linguagem guardam o conhecimento nos próprios pesos, o que traz três incômodos. O
conhecimento fica congelado no momento do treino e é caro de atualizar, o modelo não consegue apontar
de onde tirou uma informação, e tarefas que exigem fatos específicos sofrem com alucinação. Os
autores buscavam uma forma de dar ao modelo uma memória externa, consultável e fácil de atualizar.

## Ideia central

O trabalho propõe a Retrieval-Augmented Generation, que combina uma memória paramétrica, o modelo
gerador, com uma memória não paramétrica, um índice vetorial de documentos. A novidade é treinar as
duas partes juntas, de modo que o modelo aprenda a recuperar trechos relevantes e a gerar a resposta
condicionada a eles, unindo o que o gerador sabe com o que a base de documentos fornece.

## Método em linhas gerais

O sistema tem um recuperador e um gerador. O recuperador usa o Dense Passage Retrieval para
representar a pergunta e os documentos como vetores densos e buscar os trechos mais próximos em um
índice da Wikipédia. O gerador, um modelo seq2seq baseado no BART, produz a resposta a partir da
pergunta e dos trechos recuperados. Os autores apresentam duas variantes, RAG-Sequence e RAG-Token,
conforme os documentos recuperados são usados para a resposta inteira ou token a token, e treinam o
conjunto de ponta a ponta, sem supervisão direta de qual documento recuperar.

## Resultados principais

O RAG atingiu o estado da arte em várias tarefas de perguntas e respostas de domínio aberto, e gerou
respostas mais específicas e mais factuais que um modelo puramente paramétrico de tamanho
comparável. Um ponto prático importante é que o conhecimento pode ser atualizado trocando o índice de
documentos, sem retreinar o modelo, o que torna o sistema mais fácil de manter.

## Limitações e pontos de atenção

A qualidade da resposta depende muito da qualidade da recuperação, então um índice ruim ou uma busca
fraca derrubam o resultado. Manter e atualizar o índice tem custo, e o modelo ainda pode alucinar
quando os trechos recuperados são insuficientes ou enganosos. Avaliar a fidelidade da resposta às
fontes continua sendo um desafio em aberto.

## Conexão com a trilha

Este é o trabalho que fundamenta o Módulo 9, sobre RAG, e o Tutor do projeto final. A ideia de buscar
trechos relevantes e responder com base neles, citando a fonte e admitindo quando não encontra, vem
daqui. No repositório, construímos uma versão didática desse pipeline do zero, com TF-IDF e busca por
cosseno, e mostramos o caminho de produção com embeddings densos e bancos vetoriais.
