# Projeto M14, Assistente Educacional Multi-Agente

O projeto final da trilha, que integra tudo em uma sessão de tutoria viva: um assistente
educacional que conversa com o aluno, ensina, avalia, orienta, analisa e se adapta a cada um.
É o **Multi-Agent Educational Assistant with Learning Analytics and Long-Term Student
Modeling**, construído do zero reunindo as versões ricas das peças dos módulos anteriores.

## O que o assistente integra

| Componente | Origem na trilha |
|---|---|
| Tutor com RAG por trecho e citação da fonte | Módulo 9 |
| Calculadora segura como ferramenta | Módulo 10 |
| Roteador que decide a ação a cada mensagem (agent loop) | Módulo 10 |
| Time de agentes (Tutor, Evaluator, Mentor, Analytics) | Módulo 11 |
| Learning Analytics, engajamento, risco e relatório | Módulo 12 |
| Modelo do aluno: knowledge tracing, pré-requisitos e personalização | Módulo 13 |

## O que o assistente faz

- O aluno conversa em texto livre, e o assistente decide sozinho o que fazer, em vez de
  receber comandos pré-rotulados.
- O Tutor responde dúvidas buscando o trecho relevante no material, cita a fonte e adapta a
  profundidade da explicação ao domínio do aluno, considerando as suas preferências.
- A calculadora resolve contas com segurança, sem usar `eval`.
- O assistente propõe exercícios na dificuldade certa, corrige sozinho com tolerância
  numérica, atualiza o domínio do tema e registra o evento no Analytics.
- O Mentor olha o modelo e o analytics e sugere revisar os temas fracos ou avançar, indicando
  o próximo tema segundo um grafo de pré-requisitos.
- O modelo do aluno é salvo ao final e recarregado na sessão seguinte, garantindo a memória de
  longo prazo, e a sessão gera um relatório com engajamento, barras de domínio e risco.

## Como rodar

O assistente roda do zero, só com a biblioteca padrão do Python.

```bash
# Modo interativo: converse com o assistente no terminal
python final_assistant.py

# Sessão roteirizada de ponta a ponta, para uma visão rápida
python final_assistant.py --demo
```

No modo interativo, ele pergunta o seu nome, retoma o seu progresso se já houver um modelo
salvo, e ao sair grava o modelo atualizado e um `relatorio_sessao.md`. Digite `ajuda` para ver
o que dá para fazer, ou `sair` para encerrar.

## Usando no seu código

```python
from final_assistant import AssistenteEducacional, ModeloAluno

aluno = ModeloAluno("Ana", preferencias=["exemplos visuais"])
assistente = AssistenteEducacional(modelo=aluno)

print(assistente.responder("o que é a derivada?"))      # Tutor com RAG e citação
print(assistente.responder("28*3/4"))                    # Calculadora
print(assistente.responder("exercício de derivada"))     # propõe um exercício
print(assistente.responder("6"))                         # corrige e atualiza o domínio
print(assistente.responder("relatório"))                 # Analytics + Mentor

# A sessão interativa aceita entradas injetáveis, o que a torna fácil de testar.
assistente.sessao_interativa(entradas=["o que é a integral?", "sair"])
```

## Testes

Os testes rodam só com a biblioteca padrão e exercitam o fluxo de ponta a ponta: roteamento do
agente, RAG com citação, calculadora, banco de exercícios com correção, pré-requisitos,
analytics com risco, persistência e a personalização que adapta a explicação ao domínio.

```bash
pytest projects/m14-final-assistant/
```

## Como estender

- Trocar a busca TF-IDF por embeddings densos e um banco vetorial (ChromaDB ou Qdrant, Módulo 9).
- Ativar a geração via Ollama, já prevista em `gerar_explicacao`, para o Tutor redigir a
  explicação a partir do contexto na profundidade recomendada (Módulos 7 e 8).
- Orquestrar o roteamento e os agentes com LangGraph, modelando o fluxo como um grafo (Módulos 10 e 11).
- Estimar o risco de desengajamento com a regressão logística do dashboard do Módulo 12.
- Estimar o domínio com deep knowledge tracing, usando as redes do Módulo 5.

## Decisões de projeto

O assistente foi escrito do zero, sem dependências, para que toda a lógica de integração fique
visível e rode em qualquer lugar. O coordenador deixou de ser um `switch` sobre um tipo de
mensagem pré-rotulado e passou a ser um agente que roteia a mensagem livre do aluno, no espírito
do Módulo 10. Cada componente é uma versão compacta do que foi construído no módulo
correspondente, conectada pelo roteador e por um modelo do aluno compartilhado e persistente. O
resultado é um acompanhamento personalizado que emerge da integração, não de uma peça única, e
que serve de base sólida para evoluções de produção.
