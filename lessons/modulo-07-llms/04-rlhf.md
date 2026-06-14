# Aula 4, RLHF

> Esta aula fecha o módulo com o RLHF, o aprendizado por reforço com feedback humano,
> a técnica que alinha o modelo ao que as pessoas realmente preferem. Vamos entender
> as suas etapas e construir, do zero, um pequeno modelo de recompensa que aprende a
> ranquear respostas a partir de preferências.

O ajuste por instrução, da aula anterior, ensina o modelo a seguir pedidos. Mas seguir
pedidos não basta, queremos respostas que as pessoas considerem boas, úteis, honestas e
seguras. O problema é que essas qualidades são difíceis de definir com uma regra ou uma
fórmula. É mais fácil para um humano dizer qual de duas respostas é melhor do que escrever
o que torna uma resposta boa.

O RLHF, peça central do trabalho do InstructGPT, de Ouyang e colegas, transforma essas
comparações humanas em sinal de treino. Em vez de pedir a resposta perfeita, coletamos
julgamentos do tipo a resposta A é melhor que a B, treinamos um modelo de recompensa que
aprende a prever essas preferências, e então ajustamos o LLM para maximizar essa
recompensa. Nesta aula você vai entender esse fluxo e construir o coração dele, o modelo de
recompensa.

---

## Objetivos

Ao final desta aula, você deve ser capaz de:

- Explicar por que alinhar um modelo a preferências é difícil de fazer com regras.
- Descrever as três etapas do RLHF.
- Implementar um modelo de recompensa simples a partir de comparações.
- Conhecer a alternativa do DPO ao RLHF clássico.

## Teoria

O RLHF clássico tem três etapas. A primeira é o ajuste supervisionado, em que o modelo
aprende com bons exemplos de resposta, como vimos no ajuste por instrução. A segunda é o
modelo de recompensa, treinado a partir de comparações humanas, em que pessoas veem várias
respostas a um mesmo pedido e dizem qual preferem, e o modelo aprende a atribuir uma
pontuação que reproduza essas preferências. A terceira é o ajuste por reforço, em que o LLM
é treinado para gerar respostas que maximizem a pontuação do modelo de recompensa, em geral
com o algoritmo PPO.

```mermaid
flowchart LR
    S[Ajuste supervisionado] --> R[Modelo de recompensa<br/>aprende preferências]
    R --> P[Ajuste por reforço<br/>maximiza a recompensa]
    P --> A[Modelo alinhado]
```

A ideia de aprender a partir de preferências humanas, em vez de recompensas escritas à mão,
vem de Christiano e colegas, e foi o que tornou o RLHF viável para linguagem. Mais
recentemente, surgiu uma alternativa mais simples, o DPO, de Rafailov e colegas, que dispensa
treinar um modelo de recompensa separado e otimiza o LLM diretamente a partir das
comparações, simplificando bastante o processo sem perder qualidade.

## Explicação Intuitiva

Imagine treinar um cozinheiro sem nunca conseguir escrever a receita do prato perfeito. É
difícil dizer com precisão o que faz um prato ser bom, mas é fácil provar dois e dizer qual
você prefere. O RLHF usa exatamente isso. As pessoas provam pares de respostas e apontam a
melhor, e dessas escolhas o sistema destila uma noção de qualidade, o modelo de recompensa,
que passa a servir de juiz automático.

Com esse juiz em mãos, o cozinheiro, que é o LLM, treina para agradá-lo, ajustando o seu
estilo para produzir pratos que o juiz aprova. Como o juiz foi moldado pelas preferências
humanas, agradá-lo significa, na prática, agradar as pessoas. É assim que o modelo passa de
apenas seguir instruções para responder de um jeito que realmente consideramos bom.

## Explicação Matemática

O modelo de recompensa aprende a partir de pares de respostas com uma preferência humana. Se
$A$ é a resposta preferida e $B$ a preterida, queremos que a recompensa de $A$ seja maior que
a de $B$. Uma forma comum, baseada no modelo de Bradley-Terry, modela a probabilidade de $A$
ser preferida como uma função logística da diferença de recompensas:

$$
P(A \succ B) = \sigma\big(r(A) - r(B)\big),
$$

em que $r$ é a função de recompensa que estamos aprendendo e $\sigma$ é a sigmoide. Treinamos
$r$ para maximizar a probabilidade das preferências observadas, o que equivale a uma
regressão logística sobre a diferença de características das respostas. Depois, o ajuste por
reforço otimiza o LLM para produzir respostas de alta recompensa, com um termo que o impede de
se afastar demais do modelo inicial, preservando a fluência.

## Exemplo Prático

Vamos construir o coração do RLHF, o modelo de recompensa, do zero. Cada resposta é descrita
por algumas características, digamos clareza, correção e gentileza, e um humano simulado
prefere a resposta de maior qualidade verdadeira, uma combinação dessas características que o
modelo não conhece. A partir de muitos pares com preferências, treinamos um modelo logístico
que aprende os pesos da recompensa.

A expectativa, que vamos confirmar, é que o modelo recupere as proporções dos pesos
verdadeiros e passe a prever as preferências com alta acurácia, mostrando que ele aprendeu o
que torna uma resposta preferível. O código está no notebook
[notebooks/modulo-07/04-rlhf.ipynb](../../notebooks/modulo-07/04-rlhf.ipynb), então abra-o ao
lado para acompanhar.

## Código Comentado

```python
import numpy as np

rng = np.random.default_rng(0)

# Qualidade verdadeira de uma resposta = combinação das suas características.
# O modelo de recompensa não conhece estes pesos, vai tentar descobri-los.
qualidade_real = np.array([1.0, 1.5, 0.5])   # clareza, correção, gentileza

n = 400
A = rng.uniform(0, 1, (n, 3))                # características das respostas A
B = rng.uniform(0, 1, (n, 3))                # características das respostas B
# Preferência humana: 1 se A é melhor que B segundo a qualidade real, com ruído.
pref = ((A @ qualidade_real + rng.normal(0, 0.1, n)) > (B @ qualidade_real)).astype(float)

# Modelo de recompensa: aprende pesos w por regressão logística sobre A - B.
w = np.zeros(3)
for _ in range(2000):
    dif = A - B
    p = 1 / (1 + np.exp(-(dif @ w)))
    w -= 0.5 * dif.T @ (p - pref) / n

# Avalia se o modelo prevê corretamente a preferência humana.
dif = A - B
pred = (1 / (1 + np.exp(-(dif @ w))) > 0.5).astype(float)
print("Pesos verdadeiros   :", qualidade_real)
print("Pesos aprendidos     :", np.round(w, 2))
print("Acurácia nas preferências:", round(float((pred == pref).mean()), 3))
```

Ao rodar, os pesos aprendidos saem em uma escala maior que os verdadeiros, o que é normal na
regressão logística, mas guardam as mesmas proporções, a característica de peso 1,5 continua
sendo a mais importante, e a de 0,5, a menos. E o modelo prevê a preferência humana com
acurácia em torno de 0,95. Ou seja, a partir apenas de comparações do tipo esta é melhor que
aquela, ele destilou uma noção de qualidade. É esse juiz aprendido que, no RLHF completo,
guia o ajuste do LLM rumo a respostas que as pessoas preferem.

## Exercícios

1) Conceitual: Por que é mais fácil para um humano comparar duas respostas do que escrever a
   regra do que torna uma resposta boa?
2) Conceitual: Descreva as três etapas do RLHF e o papel de cada uma.
3) Prático: Mude os pesos da qualidade verdadeira e veja se o modelo de recompensa continua
   recuperando as proporções.
4) Prático: Aumente o ruído nas preferências e observe o efeito sobre a acurácia do modelo de
   recompensa.
5) Extensão: Pesquise o DPO e explique, em um parágrafo, como ele dispensa o modelo de
   recompensa separado.

## Projeto da Aula

Construa e avalie um modelo de recompensa mais rico. A entrega é um experimento que treina o
modelo de recompensa desta aula com mais características e mais pares de preferência, e mede a
sua acurácia, investigando como a quantidade de comparações afeta a qualidade do juiz
aprendido.

Considere o projeto pronto quando você tiver a curva de acurácia do modelo de recompensa em
função do número de comparações e um parágrafo discutindo por que mais dados de preferência
ajudam, e quais cuidados éticos surgem ao decidir quais preferências treinar. Com o RLHF
entendido, você fecha o módulo de LLMs e fica pronto para o Módulo 8, em que aprendemos a
extrair o melhor desses modelos com prompt engineering.

## Leituras Recomendadas

- O artigo do InstructGPT, de Ouyang e colegas, que detalha o RLHF aplicado a LLMs.
- O artigo de Christiano e colegas, sobre aprendizado por reforço a partir de preferências
  humanas.
- O artigo do DPO, de Rafailov e colegas, sobre a alternativa direta ao RLHF clássico.

## Referências Científicas

As referências abaixo são reais e estão registradas em
[references/referencias.bib](../../references/referencias.bib). As chaves entre
parênteses são as do BibTeX.

- Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human
  Feedback. NeurIPS. (`ouyang2022instructgpt`)
- Christiano, P. F., et al. (2017). Deep Reinforcement Learning from Human Preferences.
  NeurIPS. (`christiano2017rlhf`)
- Rafailov, R., et al. (2023). Direct Preference Optimization. NeurIPS. (`rafailov2023dpo`)
