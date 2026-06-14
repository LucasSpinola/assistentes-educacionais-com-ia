# Como contribuir

Obrigado pelo interesse em contribuir com o AI for Educational Assistants. Este
material é aberto e cresce melhor com a ajuda de estudantes, professores,
pesquisadores e desenvolvedores.

## Princípios do projeto

- O conteúdo precisa ser tecnicamente correto, acadêmico e didático ao mesmo tempo.
- Toda referência científica precisa ser real e verificada no Google Scholar, no
  arXiv ou pelo DOI. Não inventamos citações em hipótese alguma.
- Os exemplos rodam por padrão com modelos locais via Ollama, e oferecem um
  caminho alternativo com APIs comerciais sempre que possível.
- A escrita segue o português natural do Brasil, com texto fluido e sem excesso de
  frases curtas.

## Tipos de contribuição

- Correções de erros técnicos, de português ou de código.
- Melhorias em explicações, diagramas e exercícios.
- Novos exemplos, notebooks ou projetos.
- Tradução de termos e melhoria do glossário.
- Sugestões de leituras e referências, sempre verificadas.

## Padrão das aulas

Cada aula em `lessons/` segue o gabarito de
[docs/templates/TEMPLATE_AULA.md](docs/templates/TEMPLATE_AULA.md), com as dez
seções padronizadas, pelo menos um diagrama Mermaid e exemplos em Python. Ao
propor uma aula nova, mantenha essa estrutura.

## Padrão de código

- Python 3.11 ou superior.
- Código comentado e legível, priorizando clareza didática sobre micro-otimização.
- Quando fizer sentido, inclua um teste mínimo com `pytest`.
- Notebooks devem rodar de ponta a ponta. Limpe as saídas pesadas antes de enviar.

## Fluxo de trabalho

1. Faça um fork do repositório e crie uma branch descritiva.
2. Faça suas alterações seguindo os padrões acima.
3. Descreva claramente o que mudou e por quê na descrição do pull request.
4. Se a contribuição adiciona referências, inclua as entradas correspondentes em
   `references/referencias.bib`.

## Dúvidas

Abra uma issue descrevendo a dúvida ou a sugestão. Issues bem descritas ajudam a
manter o material consistente e de qualidade.
