# Attention Is All You Need

Referência: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L.,
e Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems
(NeurIPS). arXiv:1706.03762, https://arxiv.org/abs/1706.03762. Chave no BibTeX:
`vaswani2017attention`.

## Problema que ataca

Até 2017, os melhores modelos de tradução e de processamento de sequências eram redes recorrentes,
como LSTM e GRU, às vezes combinadas com convoluções. A recorrência processa a sequência token a
token, o que limita a paralelização no treino e dificulta capturar dependências entre palavras
distantes, porque o sinal precisa atravessar muitos passos. Os autores queriam uma arquitetura que
modelasse essas dependências de forma direta e que treinasse muito mais rápido.

## Ideia central

A proposta é o Transformer, uma arquitetura que abre mão por completo da recorrência e da convolução,
e se apoia apenas em mecanismos de atenção. A novidade está em mostrar que a self-attention, em que
cada posição da sequência olha diretamente para todas as outras, basta para modelar a linguagem, com
a vantagem de conectar quaisquer duas posições em um único passo e de permitir o processamento
paralelo de toda a sequência.

## Método em linhas gerais

O modelo segue o esquema encoder-decoder. A peça fundamental é a atenção por produto escalar
escalonado, organizada em várias cabeças (multi-head attention), que permitem ao modelo atender a
diferentes tipos de relação ao mesmo tempo. Como não há recorrência, a ordem das palavras entra por
codificações posicionais somadas aos embeddings. Cada camada combina atenção e uma rede
feed-forward, com conexões residuais e normalização, o que estabiliza o treino de modelos profundos.

## Resultados principais

O Transformer alcançou o estado da arte em tradução nas tarefas WMT 2014 inglês para alemão e inglês
para francês, superando os modelos recorrentes e convolucionais anteriores, e a um custo de treino
bem menor, graças à paralelização. O trabalho também mostrou que a arquitetura generaliza para outras
tarefas, sinalizando que a atenção seria uma base ampla, não um truque específico de tradução.

## Limitações e pontos de atenção

A self-attention tem custo que cresce com o quadrado do comprimento da sequência, o que encarece
contextos longos e motivou muita pesquisa posterior em atenção eficiente. As codificações posicionais
fixas têm limites de extrapolação para sequências mais longas que as vistas no treino, e os ganhos
plenos da arquitetura aparecem com bastante dado e computação.

## Conexão com a trilha

Este é o trabalho que fundamenta o Módulo 6, sobre Transformers, e por tabela todo o restante da
trilha. A self-attention que implementamos do zero naquele módulo vem daqui, e a arquitetura é a base
dos LLMs do Módulo 7, do RAG do Módulo 9 e dos agentes que sustentam o assistente final. Entender
este paper é entender a peça central da IA de linguagem moderna.
