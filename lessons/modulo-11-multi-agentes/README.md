# Módulo 11, Multi-Agentes

No Módulo 10, construímos um agente que faz tudo sozinho. Este módulo mostra que, muitas vezes,
é melhor dividir o trabalho entre vários agentes especializados que cooperam. Um time bem
coordenado de especialistas supera um único agente generalista, porque cada um pode ser profundo
no seu papel. É o princípio antigo da sociedade da mente, agora concretizado com agentes de LLM.

A trilha é curta e direta. Primeiro, os agentes aprendem a se comunicar, por um protocolo de
mensagens. Depois, um coordenador passa a reger o time, decidindo quem faz o quê. Por fim, cada
agente ganha uma especialidade, e montamos o time educacional completo, Tutor, Evaluator, Mentor e
Analytics. Cada aula implementa as ideias do zero, com Python puro, e no fim tudo se junta no
projeto, um sistema multi-agente que acompanha o estudo de um aluno.

## As três aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [Comunicação entre agentes](01-comunicacao-entre-agentes.md) | Define um protocolo e faz agentes conversarem |
| 2 | [Coordenação](02-coordenacao.md) | Constrói um coordenador que roteia tarefas |
| 3 | [Especialização](03-especializacao.md) | Monta o time de quatro agentes especializados |

A primeira aula trata da comunicação, a segunda da organização, e a terceira da especialização,
onde mora o projeto integrador.

## O que você leva deste módulo

Ao terminar, você sabe projetar um sistema multi-agente, com um protocolo de comunicação, um
coordenador e agentes especializados que cooperam. Entende por que a especialização e a
coordenação produzem um comportamento maior que a soma das partes, e tem um time de agentes
educacionais no portfólio. Esse sistema é a espinha dorsal do projeto final da trilha, no Módulo
14.

## Projeto do módulo

O projeto integrador está na pasta
[projects/m11-multi-agent/](../../projects/m11-multi-agent/), um time com Tutor, Evaluator, Mentor
e Analytics, coordenados por um supervisor, que cooperam para acompanhar o estudo de um aluno em
uma sessão. A avaliação alimenta a análise, e a análise orienta a recomendação. Ele roda do zero,
sem dependências, e tem testes.

## Pré-requisitos

O Módulo 10 sobre agentes é a base direta. Os exemplos e o projeto rodam só com a biblioteca
padrão do Python. As evoluções com o LLM como gerador ou coordenador usam o Ollama, conforme o
[docs/setup.md](../../docs/setup.md).

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-11/](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-11),
  um para cada aula.
- Projeto completo em [projects/m11-multi-agent/](../../projects/m11-multi-agent/).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
