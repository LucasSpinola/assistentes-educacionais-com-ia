"""Testes mínimos do time de agentes educacionais.

Rodam sem dependências externas, exercitando cada agente especializado, a
colaboração entre Evaluator e Analytics, a sugestão do Mentor e o roteamento do
Coordenador. Execute com: pytest projects/m11-multi-agent/
"""

from multi_agent import (
    Coordenador, Mensagem, TutorAgent, EvaluatorAgent, AnalyticsAgent, MentorAgent,
)


def test_tutor_explica():
    tutor = TutorAgent()
    assert "derivada" in tutor.explicar("derivada").lower()


def test_evaluator_julga():
    ev = EvaluatorAgent()
    assert ev.avaliar("21", "21")[0] is True
    assert ev.avaliar("20", "21")[0] is False


def test_analytics_agrega():
    an = AnalyticsAgent()
    an.registrar("derivada", True)
    an.registrar("derivada", False)
    an.registrar("matriz", True)
    rel = an.relatorio()
    assert rel["total"] == 3
    assert rel["acuracia"] == 0.67
    assert "derivada" in rel["temas_fracos"]


def test_mentor_sugere_revisar_ou_avancar():
    mentor = MentorAgent()
    assert "revisar" in mentor.sugerir({"total": 3, "acuracia": 0.4, "temas_fracos": ["matriz"]}).lower()
    assert "avançar" in mentor.sugerir({"total": 3, "acuracia": 0.9, "temas_fracos": []}).lower()


def test_coordenador_roteia():
    coord = Coordenador()
    assert "derivada" in coord.coordenar(Mensagem("aluno", "duvida", {"tema": "derivada"})).lower()
    assert "não sei tratar" in coord.coordenar(Mensagem("aluno", "xpto", {})).lower()


def test_sessao_completa_acompanha():
    coord = Coordenador()
    # Dois exercícios: um certo, um errado -> acurácia 0.5, mentor deve sugerir revisar.
    coord.coordenar(Mensagem("aluno", "resposta", {"tema": "derivada", "resposta": "21", "esperado": "21"}))
    coord.coordenar(Mensagem("aluno", "resposta", {"tema": "matriz", "resposta": "x", "esperado": "y"}))
    rel = coord.analytics.relatorio()
    assert rel["acuracia"] == 0.5
    sugestao = coord.coordenar(Mensagem("aluno", "pedir_sugestao", {}))
    assert "revisar" in sugestao.lower()
    assert "matriz" in sugestao.lower()
