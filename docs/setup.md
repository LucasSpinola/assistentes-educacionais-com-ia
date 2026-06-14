# Guia de Instalação

Este guia cobre tudo o que você precisa para rodar os exemplos do repositório. O
caminho padrão usa modelos locais com Ollama, que é gratuito e não exige chave de
API. As APIs comerciais são opcionais.

## 1. Python

Instale o Python 3.11 ou superior. Confirme a versão com:

```bash
python --version
```

Crie e ative um ambiente virtual na raiz do projeto:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux ou macOS
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Alguns pacotes de NLP precisam baixar dados extras na primeira vez. Por exemplo,
o spaCy precisa de um modelo de idioma e o nltk precisa de alguns recursos:

```bash
python -m spacy download pt_core_news_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## 2. Ollama, o caminho local padrão

O Ollama roda LLMs no seu próprio computador.

1. Baixe e instale o Ollama em https://ollama.com.
2. Verifique a instalação:

```bash
ollama --version
```

3. Baixe um modelo. Uma boa opção inicial é o Llama 3.1:

```bash
ollama pull llama3.1
```

4. Teste rapidamente:

```bash
ollama run llama3.1 "Explique em uma frase o que é aprendizado de máquina."
```

O serviço do Ollama costuma ficar disponível em `http://localhost:11434`, que é o
valor padrão no `.env.example`. Modelos maiores pedem mais memória, então comece
por modelos menores se o seu hardware for limitado.

## 3. APIs comerciais, opcionais

Se preferir usar OpenAI ou Anthropic em algum exemplo, copie o arquivo de
ambiente e preencha as chaves:

```bash
cp .env.example .env
```

Depois descomente os pacotes correspondentes no `requirements.txt` e instale-os.
Nenhum exemplo depende exclusivamente de API paga, sempre há a alternativa local.

## 4. Docker para o Qdrant, só no módulo avançado de RAG

O Qdrant aparece no módulo avançado de RAG como opção de produção. A forma mais
simples de subir é via Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

O ChromaDB, usado como padrão didático, não precisa de Docker, pois roda embarcado
no próprio processo do Python.

## 5. Jupyter

Para abrir os notebooks:

```bash
jupyter notebook
```

ou use a extensão de Jupyter do seu editor. Os notebooks ficam em `notebooks/`,
organizados por módulo.

## Problemas comuns

- Se a ativação do ambiente virtual falhar no PowerShell, pode ser necessário
  ajustar a política de execução com `Set-ExecutionPolicy -Scope CurrentUser
  RemoteSigned`.
- Se o `torch` não instalar pela versão de CPU ou GPU, consulte as instruções
  oficiais em https://pytorch.org para a combinação certa de sistema e CUDA.
- Se o Ollama não responder, confirme que o serviço está rodando e que a porta
  11434 não está bloqueada.
