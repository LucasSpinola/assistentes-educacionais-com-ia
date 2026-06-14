# Módulo 14, Projeto Final

Aqui tudo se junta. Ao longo de treze módulos você construiu cada peça de um assistente educacional
inteligente, do tokenizador ao knowledge tracing, quase sempre do zero. Este módulo costura essas
peças em um único sistema coeso, o Multi-Agent Educational Assistant with Learning Analytics and
Long-Term Student Modeling, um assistente que ensina, avalia, orienta, analisa e se adapta a cada
aluno. É o ponto de chegada da trilha e, ao mesmo tempo, uma base sólida para os seus próximos
projetos.

O módulo é mais enxuto que os anteriores de propósito, porque o trabalho pesado já foi feito. Em vez
de novas teorias, ele traz dois guias que mostram como pensar e montar a integração, e o projeto
final completo, pronto para estudar, rodar e estender. Primeiro desenhamos a arquitetura, decidindo
quem faz o quê e como as peças conversam. Depois construímos e avaliamos o sistema de ponta a ponta,
vendo uma sessão real do aluno em que a explicação muda de profundidade conforme o domínio dele
evolui.

## Os dois guias

| Guia | Tema | O que você leva |
|---|---|---|
| 1 | [Arquitetura do assistente](01-arquitetura.md) | O desenho do sistema e a origem de cada peça na trilha |
| 2 | [Construção e avaliação](02-construcao-e-avaliacao.md) | Como montar a integração e testá-la de ponta a ponta |

O primeiro guia desenha o sistema antes de programar, e o segundo o coloca de pé e o avalia, fechando
a trilha com o assistente rodando.

## O que você leva deste módulo

Ao terminar, você terá um assistente educacional completo no portfólio, e, mais do que isso, o
entendimento de cada peça que o forma, sem caixas-pretas. Você verá na prática como o RAG, os
agentes, o analytics e o modelo do aluno se combinam em um comportamento de acompanhamento
personalizado que nenhuma peça teria sozinha. Esse é o conhecimento que permite pesquisar, melhorar e
criar os assistentes educacionais do futuro.

## Projeto do módulo

O projeto final está na pasta
[projects/m14-final-assistant/](../../projects/m14-final-assistant/), o assistente integrado que
reúne o Tutor com RAG do Módulo 9, a calculadora segura do Módulo 10, o time de agentes coordenados do
Módulo 11, o Learning Analytics do Módulo 12 e o modelo do aluno com knowledge tracing do Módulo 13.
Um coordenador despacha as mensagens do aluno para a peça certa, e um modelo do aluno compartilhado
costura a personalização ao longo da sessão. Ele roda do zero, sem dependências, e tem testes de
integração que exercitam o fluxo completo, incluindo a personalização que adapta a profundidade da
explicação.

## Pré-requisitos

Este é o módulo que fecha a trilha, então ele se apoia em tudo o que veio antes, em especial nos
Módulos 9 (RAG), 11 (multi-agentes), 12 (analytics) e 13 (modelagem do aluno). O projeto e os testes
rodam só com a biblioteca padrão do Python. Os caminhos de evolução, como embeddings densos, banco
vetorial e geração com LLM, usam bibliotecas e o Ollama, conforme o
[docs/setup.md](../../docs/setup.md).

## Material de apoio

- Notebook executável em [notebooks/modulo-14/](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-14), que roda uma sessão
  completa do assistente.
- Projeto final completo em [projects/m14-final-assistant/](../../projects/m14-final-assistant/).
- Referências dos guias reunidas em
  [references/referencias.bib](../../references/referencias.bib).
