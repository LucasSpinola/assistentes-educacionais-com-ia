# Módulo 13, Long-Term Student Modeling

O assistente que construímos já ensina, age e analisa, mas ainda trata todos os alunos da mesma
forma. Este módulo dá a ele o que falta para personalizar de verdade, um modelo de longo prazo de
cada aluno, que cresce a cada interação e guia um ensino feito sob medida. É o que separa um
assistente genérico de um tutor que conhece o seu aluno.

A trilha constrói esse modelo peça por peça. Primeiro, o perfil, que acumula o que sabemos sobre o
aluno. Depois, a persistência, que faz o perfil sobreviver entre as sessões. Em seguida, a
modelagem cognitiva, que estima o domínio de cada habilidade com fundamento, via knowledge tracing.
Por fim, a personalização, que usa tudo isso para adaptar o que ensinar e como. Cada aula implementa
as ideias do zero, e no fim elas se juntam no projeto, um sistema adaptativo de aprendizagem.

## As quatro aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Perfil do aluno](01-perfil-do-aluno.md) | Constrói um perfil que acumula sobre o aluno |
| 2 | [Memória de longo prazo](02-memoria-de-longo-prazo.md) | Persiste o perfil entre sessões |
| 3 | [Modelagem cognitiva](03-modelagem-cognitiva.md) | Estima o domínio com knowledge tracing |
| 4 | [Personalização](04-personalizacao.md) | Adapta ensino ao modelo (projeto do módulo) |

As três primeiras aulas constroem e refinam o modelo do aluno, e a quarta o coloca para trabalhar,
onde mora o projeto integrador.

## O que você leva deste módulo

Ao terminar, você sabe modelar um aluno ao longo do tempo, com perfil persistente e estimativa de
domínio fundamentada, e usar esse modelo para personalizar o ensino, mantendo o aluno na zona certa
de desafio. Esse modelo do aluno, somado às peças dos módulos anteriores, é o último componente do
assistente educacional completo do Módulo 14.

## Projeto do módulo

O projeto integrador está na pasta
[projects/m13-student-modeling/](../../projects/m13-student-modeling/), um sistema adaptativo que
mantém um modelo do aluno com domínio por habilidade, atualizado por knowledge tracing e persistido
entre sessões, e que recomenda a ação, a profundidade e a dificuldade conforme o domínio evolui. Ele
roda do zero, sem dependências, e tem testes.

## Pré-requisitos

Os Módulos 10 (memória de agentes) e 12 (analytics) ajudam bastante, e o Teorema de Bayes do Módulo
1 fundamenta o knowledge tracing. Os exemplos e o projeto rodam só com a biblioteca padrão do
Python. A versão com explicações geradas por LLM usa o Ollama, conforme o
[docs/setup.md](../../docs/setup.md).

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-13/](../../notebooks/modulo-13/),
  um para cada aula.
- Projeto completo em [projects/m13-student-modeling/](../../projects/m13-student-modeling/).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
