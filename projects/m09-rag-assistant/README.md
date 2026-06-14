# Projeto M9, Assistente sobre Documentos com RAG

Assistente educacional que responde perguntas com base em um conjunto de notas de aula,
usando Retrieval-Augmented Generation. É o projeto que fecha o Módulo 9 e reúne tudo o
que ele ensinou.

## O que o assistente faz

- Indexa notas de aula com metadados (a disciplina de cada trecho).
- Recupera os trechos mais relevantes para uma pergunta, por similaridade.
- Filtra a busca por disciplina, quando pedido.
- Monta o contexto citando as fontes e instrui o modelo a usar só elas.
- Trata a ausência: se nada relevante é encontrado, admite que não sabe, em vez de
  inventar.
- Gera a resposta com um LLM local via Ollama, com um fallback extrativo quando o
  Ollama não está disponível.

## Como rodar

O assistente roda do zero, só com a biblioteca padrão do Python, sem instalar nada.

```bash
# Demonstração de ponta a ponta
python rag_assistant.py
```

A demonstração responde a três perguntas, uma de cálculo, uma de álgebra (filtrando a
disciplina) e uma fora do material, mostrando o tratamento da ausência.

## Usando no seu código

```python
from rag_assistant import AssistenteRAG, Documento

assistente = AssistenteRAG()
assistente.indexar([
    Documento("A derivada mede a taxa de variação de uma função.", "calculo"),
    Documento("Uma matriz organiza números em linhas e colunas.", "algebra"),
])

r = assistente.responder("o que é a derivada?")
print(r.resposta)
print("Fontes:", r.fontes)
print("Encontrou no material:", r.encontrou)
```

## Testes

Os testes não dependem de Ollama nem de bibliotecas externas. Eles exercitam a
recuperação, o filtro por disciplina e o tratamento da ausência.

```bash
pytest projects/m09-rag-assistant/
```

## Como melhorar (próximos passos sugeridos)

- Trocar o embedding TF-IDF por um Sentence Transformer, para busca por sentido.
- Trocar o banco vetorial em memória pelo ChromaDB ou pelo Qdrant.
- Adicionar chunking dos documentos longos, como na Aula 2.
- Avaliar a fidelidade das respostas às fontes citadas.

## Decisões de projeto

O assistente foi escrito do zero, com TF-IDF e busca por cosseno, para que toda a lógica
de RAG fique visível e rode sem dependências. As versões com embeddings densos e bancos
vetoriais de produção aparecem como caminhos opcionais nas aulas e nos notebooks do
módulo. A remoção de stopwords, do Módulo 3, é usada para evitar casamentos espúrios por
preposições e artigos.
