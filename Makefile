# Atalhos do projeto AI for Educational Assistants.
# No Windows, use "make" via Git Bash ou WSL, ou rode os comandos manualmente.

.PHONY: help setup lint test notebooks clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make setup      cria o ambiente virtual e instala as dependências"
	@echo "  make lint       roda uma checagem rápida de estilo (se ruff estiver instalado)"
	@echo "  make test       roda os testes com pytest"
	@echo "  make notebooks  inicia o Jupyter"
	@echo "  make clean      remove caches do Python e do pytest"

setup:
	python -m venv .venv
	. .venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt
	@echo "Ambiente pronto. Ative com: source .venv/bin/activate"

lint:
	ruff check . || echo "ruff não instalado, pulando lint"

test:
	pytest -q

notebooks:
	jupyter notebook

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
