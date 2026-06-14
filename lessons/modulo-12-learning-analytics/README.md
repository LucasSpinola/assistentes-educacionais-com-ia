# Módulo 12, Learning Analytics

Cada interação do aluno com o assistente que construímos gera dados. Este módulo ensina a
transformar esses dados em entendimento e em ação, que é o campo do Learning Analytics. O objetivo
final é prático e generoso, perceber quem está aprendendo bem, quem está se afastando, e quem
corre risco de desistir, a tempo de ajudar.

A trilha vai do dado ao modelo. Começamos pela coleta estruturada de eventos, com os cuidados
éticos que ela exige. Depois calculamos métricas que resumem a aprendizagem, combinamos algumas em
um índice de engajamento, e por fim treinamos um classificador que prevê o risco de evasão. Cada
aula constrói uma peça do zero, com Python e numpy, e no fim tudo se junta no projeto, um dashboard
que mostra o engajamento e o risco de uma turma.

## As quatro aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Coleta de dados](01-coleta-de-dados.md) | Estrutura eventos de aprendizagem com ética |
| 2 | [Métricas](02-metricas.md) | Calcula acurácia, volume e progresso |
| 3 | [Engajamento](03-engajamento.md) | Combina métricas em um índice de engajamento |
| 4 | [Predição de evasão](04-predicao-de-evasao.md) | Treina um classificador de risco (projeto) |

As três primeiras aulas preparam e resumem os dados, e a quarta os usa para prever, onde mora o
dashboard que fecha o módulo.

## O que você leva deste módulo

Ao terminar, você sabe coletar dados de aprendizagem de forma estruturada e ética, calcular
métricas úteis, medir engajamento e treinar um modelo de predição de evasão, montando um dashboard
que apoia decisões pedagógicas. Esse olhar analítico, ligado às técnicas de ML do Módulo 2, é o que
o assistente final usará para acompanhar o aluno e personalizar a ajuda.

## Projeto do módulo

O projeto integrador está na pasta
[projects/m12-analytics-dashboard/](../../projects/m12-analytics-dashboard/), um dashboard que
calcula o engajamento de uma turma, treina um classificador de evasão e lista os alunos do maior
para o menor risco, destacando quem precisa de atenção. Ele roda do zero, só com numpy, e tem
testes.

## Pré-requisitos

O Módulo 2 sobre Machine Learning, em especial a regressão logística e a validação, é a base
direta do classificador de evasão. Os exemplos e o projeto rodam com numpy. As versões com pandas
e o app com Streamlit e Plotly entram com o ambiente de [docs/setup.md](../../docs/setup.md).

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-12/](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-12),
  um para cada aula.
- Projeto completo em [projects/m12-analytics-dashboard/](../../projects/m12-analytics-dashboard/).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
