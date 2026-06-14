# Notebooks

Notebooks Jupyter executáveis, organizados por módulo, espelhando as aulas em
[lessons/](../lessons/). Cada notebook traz o código comentado da aula de forma
interativa, para o aluno rodar e experimentar.

## Notebooks por módulo

| Módulo | Pasta | Tema |
|---|---|---|
| 01 | [modulo-01](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-01) | Introdução à IA |
| 02 | [modulo-02](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-02) | Fundamentos de Machine Learning |
| 03 | [modulo-03](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-03) | Fundamentos de NLP |
| 04 | [modulo-04](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-04) | Word Embeddings |
| 05 | [modulo-05](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-05) | Deep Learning para NLP |
| 06 | [modulo-06](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-06) | Transformers |
| 07 | [modulo-07](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-07) | Large Language Models |
| 08 | [modulo-08](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-08) | Prompt Engineering |
| 09 | [modulo-09](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-09) | Retrieval-Augmented Generation |
| 10 | [modulo-10](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-10) | Agentes |
| 11 | [modulo-11](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-11) | Multi-Agentes |
| 12 | [modulo-12](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-12) | Learning Analytics |
| 13 | [modulo-13](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-13) | Long-Term Student Modeling |
| 14 | [modulo-14](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-14) | Projeto Final |

## Convenções

- Uma pasta por módulo, no formato `modulo-01/`, `modulo-02/` e assim por diante.
- O nome de cada notebook acompanha o número e o tema da aula, por exemplo
  `01-tokenizacao.ipynb`.
- Os notebooks rodam de ponta a ponta com o ambiente de `requirements.txt` e, por
  padrão, com modelos locais via Ollama.
- Células que dependem de API comercial são opcionais e devem degradar de forma
  graciosa quando não houver chave configurada.
- Antes de versionar, limpe as saídas pesadas para manter o repositório leve.

## Como rodar

Ative o ambiente virtual, instale as dependências conforme `docs/setup.md` e
inicie o Jupyter:

```bash
jupyter notebook
```
