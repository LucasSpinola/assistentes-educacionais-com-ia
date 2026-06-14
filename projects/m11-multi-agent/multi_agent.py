"""Time de agentes educacionais, projeto do Módulo 11.

Quatro agentes especializados, coordenados por um supervisor, cooperam para
acompanhar o estudo de um aluno:
- TutorAgent: ensina, explicando conceitos a partir de uma base de material;
- EvaluatorAgent: avalia as respostas do aluno e dá feedback;
- AnalyticsAgent: acumula os eventos da sessão e mede o progresso;
- MentorAgent: olha o desempenho e sugere revisar ou avançar.

Tudo roda do zero, só com a biblioteca padrão do Python. O fluxo é coordenado por
mensagens tipadas, e o LLM (Ollama) pode ser acrescentado depois como gerador.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Mensagem:
    """Protocolo de comunicação entre os agentes."""

    remetente: str
    tipo: str          # "duvida", "resposta", "pedir_sugestao", "pedir_relatorio"
    conteudo: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agentes especializados
# ---------------------------------------------------------------------------
class TutorAgent:
    nome = "tutor"

    def __init__(self, material: dict[str, str] | None = None):
        self.material = material or {
            "derivada": "A derivada mede a taxa de variação instantânea de uma função.",
            "matriz": "Uma matriz organiza números em linhas e colunas.",
            "integral": "A integral acumula valores e é a inversa da derivada.",
        }

    def explicar(self, tema: str) -> str:
        return self.material.get(tema, f"Vou explicar {tema} com calma, passo a passo.")


class EvaluatorAgent:
    nome = "evaluator"

    def avaliar(self, resposta, esperado) -> tuple[bool, str]:
        correto = str(resposta).strip().lower() == str(esperado).strip().lower()
        feedback = "Correto, muito bem!" if correto else f"Ainda não. A resposta era {esperado}."
        return correto, feedback


class AnalyticsAgent:
    nome = "analytics"

    def __init__(self):
        self.eventos: list[tuple[str, bool]] = []

    def registrar(self, tema: str, correto: bool) -> None:
        self.eventos.append((tema, correto))

    def acuracia(self) -> float:
        if not self.eventos:
            return 0.0
        return sum(1 for _, c in self.eventos if c) / len(self.eventos)

    def temas_fracos(self) -> list[str]:
        """Temas em que o aluno errou ao menos uma vez."""
        return sorted({tema for tema, correto in self.eventos if not correto})

    def relatorio(self) -> dict:
        return {
            "total": len(self.eventos),
            "acuracia": round(self.acuracia(), 2),
            "temas_fracos": self.temas_fracos(),
        }


class MentorAgent:
    nome = "mentor"

    def __init__(self, limiar: float = 0.6):
        self.limiar = limiar

    def sugerir(self, relatorio: dict) -> str:
        if relatorio["total"] == 0:
            return "Vamos começar com alguns exercícios para eu te conhecer."
        if relatorio["acuracia"] < self.limiar:
            fracos = ", ".join(relatorio["temas_fracos"]) or "o tema atual"
            return f"Sugiro revisar antes de avançar, com foco em: {fracos}."
        return "Bom desempenho! Pode avançar para o próximo tema."


# ---------------------------------------------------------------------------
# Coordenador (supervisor)
# ---------------------------------------------------------------------------
class Coordenador:
    """Roteia mensagens e orquestra a colaboração entre os agentes."""

    def __init__(self):
        self.tutor = TutorAgent()
        self.evaluator = EvaluatorAgent()
        self.analytics = AnalyticsAgent()
        self.mentor = MentorAgent()

    def coordenar(self, mensagem: Mensagem) -> str:
        if mensagem.tipo == "duvida":
            return self.tutor.explicar(mensagem.conteudo.get("tema", ""))

        if mensagem.tipo == "resposta":
            # Evaluator avalia e Analytics registra: dois agentes colaborando.
            correto, feedback = self.evaluator.avaliar(
                mensagem.conteudo.get("resposta"), mensagem.conteudo.get("esperado")
            )
            self.analytics.registrar(mensagem.conteudo.get("tema", "geral"), correto)
            return feedback

        if mensagem.tipo == "pedir_relatorio":
            return str(self.analytics.relatorio())

        if mensagem.tipo == "pedir_sugestao":
            return self.mentor.sugerir(self.analytics.relatorio())

        return f"Não sei tratar mensagens do tipo '{mensagem.tipo}'."


def sessao_exemplo() -> None:
    """Simula uma sessão de estudo com o time completo cooperando."""
    coord = Coordenador()

    eventos = [
        Mensagem("aluno", "duvida", {"tema": "derivada"}),
        Mensagem("aluno", "resposta", {"tema": "derivada", "resposta": "21", "esperado": "21"}),
        Mensagem("aluno", "resposta", {"tema": "derivada", "resposta": "10", "esperado": "12"}),
        Mensagem("aluno", "resposta", {"tema": "matriz", "resposta": "x", "esperado": "y"}),
        Mensagem("aluno", "pedir_relatorio", {}),
        Mensagem("aluno", "pedir_sugestao", {}),
    ]
    for m in eventos:
        print(f"[{m.tipo}] -> {coord.coordenar(m)}")


if __name__ == "__main__":
    sessao_exemplo()
