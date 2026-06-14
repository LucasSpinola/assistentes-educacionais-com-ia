# Projeto M11, Time de Agentes Educacionais

Um time de quatro agentes especializados, coordenados por um supervisor, que cooperam para
acompanhar o estudo de um aluno. É o projeto que fecha o Módulo 11 e reúne a comunicação por
mensagens, a coordenação e a especialização que ele ensinou.

## Os agentes

- **TutorAgent**: ensina, explicando conceitos a partir de uma base de material.
- **EvaluatorAgent**: avalia as respostas do aluno e dá feedback.
- **AnalyticsAgent**: acumula os eventos da sessão e mede o progresso (acurácia, temas fracos).
- **MentorAgent**: olha o desempenho e sugere revisar (com foco nos temas fracos) ou avançar.
- **Coordenador**: roteia as mensagens e orquestra a colaboração, por exemplo fazendo a avaliação
  do Evaluator alimentar o Analytics.

## Como rodar

O time roda do zero, só com a biblioteca padrão do Python, sem instalar nada.

```bash
python multi_agent.py
```

A demonstração simula uma sessão: o aluno tira uma dúvida, responde a três exercícios que são
avaliados e registrados, pede um relatório do progresso e recebe uma sugestão do Mentor baseada no
desempenho.

## Usando no seu código

```python
from multi_agent import Coordenador, Mensagem

coord = Coordenador()
print(coord.coordenar(Mensagem("aluno", "duvida", {"tema": "derivada"})))
coord.coordenar(Mensagem("aluno", "resposta", {"tema": "derivada", "resposta": "21", "esperado": "21"}))
print(coord.coordenar(Mensagem("aluno", "pedir_sugestao", {})))
```

## Testes

Os testes não dependem de bibliotecas externas. Eles exercitam cada agente, a colaboração entre
Evaluator e Analytics, a sugestão do Mentor e o roteamento do Coordenador.

```bash
pytest projects/m11-multi-agent/
```

## Como melhorar (próximos passos sugeridos)

- Usar o LLM (Ollama) para o Tutor gerar explicações ricas e o Evaluator dar feedback detalhado.
- Trocar o roteamento por regras do Coordenador por um roteamento decidido pelo LLM.
- Ligar o Tutor ao assistente de RAG do Módulo 9 como fonte de material.
- Persistir o Analytics entre sessões, conectando com os Módulos 12 e 13.

## Decisões de projeto

Os agentes foram escritos do zero, cada um com um papel e um contrato claros, para que a
cooperação fique visível e rode sem dependências. O Coordenador centraliza o roteamento (padrão
de supervisor) e orquestra a colaboração entre agentes. A inteligência de acompanhamento, em que
a avaliação alimenta a análise e a análise orienta a recomendação, emerge da interação entre os
especialistas, não de um agente único.
