# Projeto M13, Sistema Adaptativo de Aprendizagem

Um sistema que ajusta as explicações e os exercícios ao perfil e ao histórico de cada aluno. É o
projeto que fecha o Módulo 13 e reúne o perfil persistente, o knowledge tracing e a personalização
que ele ensinou.

## O que o sistema faz

- Mantém um modelo do aluno com o domínio estimado de cada habilidade.
- Atualiza o domínio a cada resposta com knowledge tracing (considerando deslize e chute).
- Persiste o modelo entre sessões, salvando e carregando em JSON.
- Recomenda a próxima ação (revisar, praticar, avançar), a profundidade da explicação e a
  dificuldade do exercício, afinando pela preferência do aluno.
- Mantém o aluno na zona certa de desafio conforme o seu domínio evolui.

## Como rodar

O sistema roda do zero, só com a biblioteca padrão do Python.

```bash
python adaptive_system.py
```

A demonstração simula a trajetória de uma aluna que erra e depois acerta cada vez mais, e mostra a
recomendação evoluindo de revisar para praticar e avançar conforme o domínio sobe.

## Usando no seu código

```python
from adaptive_system import ModeloAluno

aluno = ModeloAluno("Ana", preferencias=["exemplos visuais"])
aluno.registrar_resposta("derivada", correto=True)
print(aluno.dominio)                      # domínio estimado por habilidade
print(aluno.recomendacao("derivada"))     # ação, profundidade, dificuldade

aluno.salvar("modelo_ana.json")           # persiste entre sessões
mesma_ana = ModeloAluno.carregar("modelo_ana.json")
```

## Testes

Os testes rodam só com a biblioteca padrão. Eles exercitam o knowledge tracing, a recomendação
adaptativa, a persistência (ida e volta fiel) e a evolução da recomendação ao longo da trajetória.

```bash
pytest projects/m13-student-modeling/
```

## Como melhorar (próximos passos sugeridos)

- Estimar o domínio com deep knowledge tracing, usando as redes recorrentes do Módulo 5.
- Usar o LLM (Ollama) para gerar as explicações na profundidade recomendada.
- Selecionar o próximo exercício de um banco real, com dificuldade calibrada.
- Integrar o modelo do aluno ao agente tutor do Módulo 10 e ao analytics do Módulo 12.

## Decisões de projeto

O sistema foi escrito do zero para que toda a lógica de modelagem e adaptação fique visível e rode
sem dependências. O knowledge tracing dá uma estimativa de domínio com fundamento e interpretação,
superior a uma média simples. A persistência em JSON garante a continuidade entre sessões, base do
acompanhamento de longo prazo. A personalização persegue a zona certa de desafio, adaptando o
ensino a cada aluno em vez de tratar todos igual.
