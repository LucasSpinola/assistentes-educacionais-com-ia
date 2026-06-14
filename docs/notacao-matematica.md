# Notação Matemática

Esta página reúne a notação usada nas seções de Explicação Matemática das aulas.
A ideia é manter os símbolos consistentes em todo o material. As fórmulas usam
LaTeX, que renderiza no GitHub e nos notebooks.

## Convenções gerais

| Símbolo | Significado |
|---|---|
| $x$ | escalar |
| $\mathbf{x}$ | vetor |
| $\mathbf{X}$ | matriz |
| $x_i$ | i-ésimo elemento de um vetor |
| $x^{(i)}$ | i-ésimo exemplo de um conjunto de dados |
| $\mathbb{R}^n$ | espaço dos vetores reais de dimensão $n$ |
| $\hat{y}$ | valor previsto pelo modelo |
| $y$ | valor real, ou rótulo |
| $\theta$ | parâmetros do modelo |
| $m$ | número de exemplos |
| $n$ | número de variáveis ou dimensões |

## Funções comuns

| Símbolo | Significado |
|---|---|
| $L(\hat{y}, y)$ | função de perda de um exemplo |
| $J(\theta)$ | função de custo sobre o conjunto |
| $\sigma(z)$ | função sigmoide, $\frac{1}{1 + e^{-z}}$ |
| $\text{softmax}(\mathbf{z})_i$ | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ |
| $\nabla_\theta J$ | gradiente do custo em relação a $\theta$ |

## Probabilidade

| Símbolo | Significado |
|---|---|
| $P(A)$ | probabilidade do evento $A$ |
| $P(A \mid B)$ | probabilidade de $A$ dado $B$ |
| $\mathbb{E}[X]$ | valor esperado de $X$ |
| $\text{Var}(X)$ | variância de $X$ |

## Álgebra linear

| Símbolo | Significado |
|---|---|
| $\mathbf{a} \cdot \mathbf{b}$ | produto escalar |
| $\mathbf{A}^\top$ | transposta da matriz $\mathbf{A}$ |
| $\lVert \mathbf{x} \rVert$ | norma do vetor $\mathbf{x}$ |
| $\cos(\mathbf{a}, \mathbf{b})$ | similaridade do cosseno entre dois vetores |

Quando uma aula introduzir um símbolo novo, ele será definido no próprio texto e,
se for de uso recorrente, acrescentado aqui.
