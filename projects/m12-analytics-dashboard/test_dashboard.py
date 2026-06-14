"""Testes mínimos do dashboard de aprendizado.

Rodam só com numpy (sem Streamlit), exercitando o índice de engajamento, o
treino e a acurácia do classificador de evasão, e a sinalização de risco.
Execute com: pytest projects/m12-analytics-dashboard/
"""

from dashboard import TurmaAnalytics, score_engajamento


def test_engajamento_separa_alunos():
    alto = score_engajamento(dias_ativos=12, total_dias=14, exercicios=18, sequencia=6)
    baixo = score_engajamento(dias_ativos=2, total_dias=14, exercicios=3, sequencia=0)
    assert alto > 0.8
    assert baixo < 0.4


def test_classificador_acima_do_acaso():
    turma = TurmaAnalytics(n=200, seed=0)
    turma.treinar()
    assert turma.acuracia_modelo() > 0.75


def test_dashboard_ordenado_por_risco():
    turma = TurmaAnalytics(n=200, seed=0)
    turma.treinar()
    linhas = turma.dashboard()
    probs = [l.prob_evasao for l in linhas]
    assert probs == sorted(probs, reverse=True)   # do maior risco para o menor


def test_resumo_sinaliza_alunos_em_risco():
    turma = TurmaAnalytics(n=200, seed=0)
    turma.treinar()
    resumo = turma.resumo()
    assert resumo["alunos"] == 200
    assert resumo["em_risco"] > 0
    assert 0 < resumo["evadiram_historico"] < 200
