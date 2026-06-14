# Módulo 1, Introdução à IA

Antes de construir qualquer assistente educacional, precisamos combinar o que
estamos chamando de Inteligência Artificial. É isso que este primeiro módulo faz.
Ele conta como a área surgiu, organiza o vocabulário que vai reaparecer ao longo
de todo o curso e, acima de tudo, apresenta as três grandes maneiras de fazer IA
que orientam a trilha inteira, a simbólica, a estatística e a generativa.

O que costura as cinco aulas é uma disputa antiga. De um lado, a aposta de que
basta escrever as regras certas para a máquina ser inteligente. De outro, a ideia
de que ela deve aprender sozinha, a partir de exemplos. A IA generativa, que fecha
o módulo, nasce desse segundo lado e leva a previsão de padrões ao ponto de criar
conteúdo novo, que é justamente o que permite a um assistente conversar de forma
natural. Em vez de ficar só na teoria, três das cinco aulas pedem que você
implemente do zero um pequeno sistema que representa cada abordagem, para sentir as
diferenças na prática.

## As cinco aulas

| Aula | Tema | O que você faz na prática |
|---|---|---|
| 1 | [O que é IA](01-o-que-e-ia.md) | Resolve a mesma tarefa com as três abordagens e compara |
| 2 | [História da IA](02-historia-da-ia.md) | Treina um perceptron e descobre por que ele falha no XOR |
| 3 | [IA simbólica](03-ia-simbolica.md) | Constrói um motor de inferência que recomenda e explica |
| 4 | [IA estatística](04-ia-estatistica.md) | Implementa um classificador Naive Bayes na mão |
| 5 | [IA generativa](05-ia-generativa.md) | Cria um gerador de texto e faz a ponte para os LLMs |

A ordem importa, porque cada aula apoia a seguinte. A primeira dá o mapa geral, a
segunda mostra como a área chegou até aqui, e as três últimas abrem uma a uma as
abordagens que aparecem no resto da trilha.

## O que você leva deste módulo

Ao terminar, você consegue explicar com clareza o que é e o que não é IA,
distinguir Machine Learning de Deep Learning, e dizer em que situação faz mais
sentido escrever regras, aprender com dados ou gerar conteúdo. Esse repertório é o
que vai te permitir, mais adiante, escolher as ferramentas certas para cada parte
de um assistente educacional.

## Projeto do módulo

Não há um projeto de software único aqui, já que cada aula traz a sua própria
entrega prática. A proposta de fechamento é conceitual, montar um mapa que ligue as
três abordagens de IA aos componentes de um assistente educacional, indicando onde
cada uma seria mais útil. Esse mapa vira uma bússola para os módulos seguintes.

## Pré-requisitos

Nenhum conhecimento prévio de IA é necessário. Para rodar os notebooks, basta
Python básico e o ambiente descrito em [docs/setup.md](../../docs/setup.md). As
partes que usam modelos generativos rodam localmente com o Ollama, sem chave de
API.

## Material de apoio

- Notebooks executáveis em [notebooks/modulo-01/](https://github.com/LucasSpinola/assistentes-educacionais-com-ia/tree/main/notebooks/modulo-01),
  um para cada aula.
- Glossário dos termos que aparecem no módulo em
  [docs/glossario.md](../../docs/glossario.md).
- Referências de cada aula reunidas em
  [references/referencias.bib](../../references/referencias.bib).
