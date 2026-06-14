"""Agente tutor, projeto do Módulo 10.

Reúne tudo o que o módulo ensinou:
- ferramentas: uma calculadora segura (sem eval) e uma busca no material (RAG, no
  estilo do Módulo 9, com TF-IDF e remoção de stopwords),
- um agent loop que decide qual ferramenta usar,
- planejamento de várias etapas para problemas encadeados,
- memória de longo prazo sobre o aluno, para personalizar o atendimento.

Roda do zero, só com a biblioteca padrão do Python. O LLM (Ollama) é opcional como
controlador e como gerador; sem ele, um controlador por regras decide a ferramenta.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Ferramenta 1: calculadora segura
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def calcular(expressao: str) -> float:
    """Avalia uma expressão aritmética com segurança, sem usar eval."""
    def ev(no):
        if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
            return no.value
        if isinstance(no, ast.BinOp) and type(no.op) in _OPS:
            return _OPS[type(no.op)](ev(no.left), ev(no.right))
        if isinstance(no, ast.UnaryOp) and type(no.op) in _OPS:
            return _OPS[type(no.op)](ev(no.operand))
        raise ValueError("expressão não permitida")
    return ev(ast.parse(str(expressao), mode="eval").body)


def extrair_expressao(texto: str) -> str:
    """Junta apenas números e operadores, ignorando o resto do texto."""
    return "".join(re.findall(r"\d+\.?\d*|[+\-*/()]", texto))


# ---------------------------------------------------------------------------
# Ferramenta 2: busca no material (RAG simples)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "de", "da", "do", "das", "dos", "em",
    "na", "no", "que", "é", "e", "com", "qual", "para", "por", "se", "como",
    "the", "of",
}


def _tok(texto: str) -> list[str]:
    return [p for p in re.findall(r"\w+", texto.lower()) if p not in _STOPWORDS]


class BaseConhecimento:
    """Busca por similaridade (TF-IDF + cosseno) sobre notas de aula."""

    def __init__(self, documentos: list[str], limiar: float = 0.05):
        self.documentos = list(documentos)
        self.limiar = limiar
        n = len(documentos)
        df: Counter = Counter()
        for d in documentos:
            for w in set(_tok(d)):
                df[w] += 1
        self.idf = {w: math.log(n / f) for w, f in df.items()}
        self.vetores = [self._vetor(d) for d in documentos]

    def _vetor(self, texto: str) -> dict:
        tf = Counter(_tok(texto))
        return {w: c * self.idf.get(w, 0.0) for w, c in tf.items()}

    @staticmethod
    def _cos(a: dict, b: dict) -> float:
        prod = sum(p * b.get(w, 0.0) for w, p in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return prod / (na * nb) if na and nb else 0.0

    def buscar(self, consulta: str) -> str:
        q = self._vetor(consulta)
        ranking = sorted(
            ((self._cos(q, self.vetores[i]), i) for i in range(len(self.documentos))),
            reverse=True,
        )
        if not ranking or ranking[0][0] < self.limiar:
            return "Não encontrei isso no material disponível."
        return self.documentos[ranking[0][1]]


# ---------------------------------------------------------------------------
# Memória de longo prazo sobre o aluno
# ---------------------------------------------------------------------------
class MemoriaAluno:
    def __init__(self):
        self._fatos: dict[str, str] = {}

    def lembrar(self, chave: str, valor: str) -> None:
        self._fatos[chave] = valor

    def recuperar(self, chave: str, padrao: str | None = None) -> str | None:
        return self._fatos.get(chave, padrao)

    def contexto(self) -> str:
        if not self._fatos:
            return "Nenhuma informação prévia sobre o aluno."
        return "; ".join(f"{c}: {v}" for c, v in self._fatos.items())


@dataclass
class Acao:
    ferramenta: str
    argumento: str
    observacao: object
    resposta: str


# ---------------------------------------------------------------------------
# O agente tutor
# ---------------------------------------------------------------------------
class AgenteTutor:
    """Agente tutor: decide ferramentas, planeja, lembra do aluno."""

    def __init__(self, base: BaseConhecimento, modelo_llm: str = "llama3.1"):
        self.base = base
        self.modelo_llm = modelo_llm
        self.memoria = MemoriaAluno()

    # ----- decisão (controlador por regras) -----------------------------
    @staticmethod
    def _eh_conta(pergunta: str) -> bool:
        return bool(re.search(r"\d", pergunta) and re.search(r"[+\-*/]", pergunta))

    def responder(self, pergunta: str) -> Acao:
        """Agent loop de uma etapa: decide a ferramenta, age e responde."""
        if self._eh_conta(pergunta):
            arg = extrair_expressao(pergunta)
            try:
                obs = calcular(arg)
                resposta = f"O resultado de {arg} é {obs}."
            except Exception as erro:
                obs, resposta = None, f"Não consegui calcular: {erro}"
            return Acao("calcular", arg, obs, resposta)

        obs = self.base.buscar(pergunta)
        nome = self.memoria.recuperar("nome")
        saudacao = f"{nome}, " if nome else ""
        resposta = f"{saudacao}{obs}"
        return Acao("buscar", pergunta, obs, resposta)

    # ----- planejamento de várias etapas --------------------------------
    def resolver_problema(self, passos: list[tuple[str, str]], limite: int = 5) -> list[Acao]:
        """Executa um plano encadeado de contas. Cada template pode usar {r}."""
        acoes: list[Acao] = []
        resultado = None
        for i, (descricao, template) in enumerate(passos):
            if i >= limite:
                break
            expr = template.format(r=resultado) if resultado is not None else template
            resultado = calcular(expr)
            acoes.append(Acao("calcular", expr, resultado, f"{descricao}: {expr} = {resultado}"))
        return acoes


# Base de conhecimento de exemplo.
DOCUMENTOS_EXEMPLO = [
    "A derivada mede a taxa de variação instantânea de uma função em um ponto.",
    "A regra da cadeia permite derivar funções compostas.",
    "Uma matriz é uma tabela de números organizada em linhas e colunas.",
    "O determinante de uma matriz indica se ela é invertível.",
]


def demo() -> None:
    base = BaseConhecimento(DOCUMENTOS_EXEMPLO)
    tutor = AgenteTutor(base)
    tutor.memoria.lembrar("nome", "Ana")
    tutor.memoria.lembrar("nivel", "iniciante em cálculo")

    print("Memória do aluno:", tutor.memoria.contexto(), "\n")

    for pergunta in ["quanto é 28*3/4 ?", "o que é a derivada?", "qual a capital da França?"]:
        acao = tutor.responder(pergunta)
        print(f"Pergunta: {pergunta}")
        print(f"  Ferramenta: {acao.ferramenta} | Resposta: {acao.resposta}\n")

    print("Problema de várias etapas (28 alunos, 3 cadernos cada, pacotes de 4):")
    for acao in tutor.resolver_problema([("Cadernos", "28*3"), ("Pacotes", "{r}/4")]):
        print("  ", acao.resposta)


if __name__ == "__main__":
    demo()
