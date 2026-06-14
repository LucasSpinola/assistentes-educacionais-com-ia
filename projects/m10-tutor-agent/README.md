# Projeto M10, Agente Tutor

Agente tutor que usa ferramentas para ajudar o aluno: uma calculadora segura para contas e
uma busca no material (RAG) para dúvidas de conteúdo. É o projeto que fecha o Módulo 10 e
reúne tudo o que ele ensinou, o agent loop, o tool calling, o planejamento de etapas e a
memória do aluno.

## O que o agente faz

- Decide, a cada pergunta, qual ferramenta usar (calculadora ou busca).
- Calcula expressões com segurança, sem usar `eval`.
- Busca no material de aula por similaridade, e admite quando não encontra.
- Planeja problemas de várias etapas, encadeando os resultados das contas.
- Lembra de informações do aluno (nome, nível) e personaliza o atendimento.

## Como rodar

O agente roda do zero, só com a biblioteca padrão do Python, sem instalar nada.

```bash
python tutor_agent.py
```

A demonstração mostra o agente resolvendo uma conta, respondendo a uma dúvida de conteúdo,
recusando uma pergunta fora do material e resolvendo um problema de duas etapas, sempre
lembrando do aluno.

## Usando no seu código

```python
from tutor_agent import AgenteTutor, BaseConhecimento, DOCUMENTOS_EXEMPLO

tutor = AgenteTutor(BaseConhecimento(DOCUMENTOS_EXEMPLO))
tutor.memoria.lembrar("nome", "Ana")

print(tutor.responder("quanto é 28*3/4 ?").resposta)
print(tutor.responder("o que é a derivada?").resposta)

# Planejamento de várias etapas
for acao in tutor.resolver_problema([("Cadernos", "28*3"), ("Pacotes", "{r}/4")]):
    print(acao.resposta)
```

## Testes

Os testes não dependem de Ollama nem de bibliotecas externas. Eles exercitam a calculadora
segura, o roteamento de ferramentas, a busca, a memória, o tratamento da ausência e o
planejamento.

```bash
pytest projects/m10-tutor-agent/
```

## Como melhorar (próximos passos sugeridos)

- Trocar o controlador por regras por um controlador com LLM, que decide a ferramenta por
  tool calling em JSON (Aula 2).
- Usar o assistente de RAG do Módulo 9 como ferramenta de busca, com embeddings densos.
- Orquestrar o agente com LangGraph, modelando o fluxo como um grafo (Aula 5).
- Persistir a memória do aluno entre execuções, ligando com o Módulo 13.

## Decisões de projeto

O agente foi escrito do zero para que toda a lógica fique visível e rode sem dependências. O
controlador por regras é o padrão executável; o LLM como controlador e gerador aparece como
opção nas aulas. A calculadora usa um avaliador seguro baseado em `ast`, sem `eval`, para
nunca executar código arbitrário. A busca reusa o RAG do Módulo 9, com remoção de stopwords.
