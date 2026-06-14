# Projeto M12, Dashboard de Aprendizado

Um dashboard de Learning Analytics que mostra o engajamento de uma turma e sinaliza os alunos
em risco de evasão. É o projeto que fecha o Módulo 12 e reúne a coleta de dados, as métricas, o
índice de engajamento e o classificador de evasão que ele ensinou.

## O que o dashboard faz

- Calcula um índice de engajamento por aluno, combinando frequência, volume e constância.
- Treina um classificador de evasão (regressão logística) sobre os dados da turma.
- Estima a probabilidade de evasão de cada aluno.
- Produz um painel que lista os alunos do maior para o menor risco, destacando os em risco, para
  o professor priorizar a atenção.
- Mostra um resumo da turma: total, evasões históricas, acurácia do modelo e número de alunos em
  risco.

## Como rodar

O dashboard roda do zero, só com numpy.

```bash
python dashboard.py
```

A demonstração gera uma turma de 200 alunos, treina o modelo e imprime o resumo e o top de alunos
em risco.

## Usando no seu código

```python
from dashboard import TurmaAnalytics

turma = TurmaAnalytics(n=200, seed=0)
turma.treinar()

print(turma.resumo())
for linha in turma.dashboard()[:5]:
    print(linha.aluno, linha.engajamento, round(linha.prob_evasao, 2), linha.em_risco)
```

## Testes

Os testes rodam só com numpy. Eles exercitam o índice de engajamento, o treino e a acurácia do
classificador, a ordenação do painel por risco e a sinalização dos alunos.

```bash
pytest projects/m12-analytics-dashboard/
```

## Como melhorar (próximos passos sugeridos)

- Apresentar o painel como um app interativo com Streamlit e gráficos com Plotly.
- Calcular as métricas a partir de eventos reais coletados (Aula 1), em vez de features sintéticas.
- Avaliar o modelo com validação cruzada (Módulo 2) e calibrar o limiar de risco.
- Acrescentar explicações de por que cada aluno foi sinalizado, ligando com a ética.

## Decisões de projeto

O dashboard foi escrito do zero, com numpy, para que a lógica de engajamento e de predição fique
visível e rode sem dependências. Os dados da turma são sintéticos, mas o pipeline é o mesmo de um
caso real, calcular métricas, treinar o classificador e priorizar os alunos. O sinalizador serve
para o professor dirigir a atenção a quem mais precisa, com apoio, nunca para rotular ou excluir o
aluno.
