"""Testes mínimos do sistema adaptativo de aprendizagem.

Rodam só com a biblioteca padrão (sem dependências externas), exercitando o
knowledge tracing, a recomendação adaptativa, a persistência e a trajetória.
Execute com: pytest projects/m13-student-modeling/
"""

import os
import tempfile

from adaptive_system import (
    ModeloAluno, atualizar_maestria, recomendar, simular_trajetoria,
)


def test_kt_sobe_com_acertos_e_fica_baixo_com_erros():
    p = 0.3
    for _ in range(5):
        p = atualizar_maestria(p, correto=True)
    assert p > 0.9                      # acertando, o domínio sobe

    q = 0.3
    for _ in range(5):
        q = atualizar_maestria(q, correto=False)
    assert q < 0.35                     # errando, fica baixo


def test_recomendar_mapeia_dominio():
    assert recomendar(0.2)["acao"] == "revisar"
    assert recomendar(0.6)["acao"] == "praticar"
    assert recomendar(0.9)["acao"] == "avançar"
    assert "diagrama" in recomendar(0.2, ["exemplos visuais"])["profundidade"]


def test_persistencia_round_trip():
    aluno = ModeloAluno("Ana", preferencias=["exemplos visuais"])
    aluno.registrar_resposta("derivada", True)
    aluno.registrar_resposta("derivada", True)
    caminho = os.path.join(tempfile.gettempdir(), "modelo_ana_teste.json")
    aluno.salvar(caminho)
    carregado = ModeloAluno.carregar(caminho)
    assert carregado.to_dict() == aluno.to_dict()
    os.remove(caminho)


def test_trajetoria_evolui_a_recomendacao():
    aluno = ModeloAluno("Bruno")
    historico = simular_trajetoria(aluno, "matriz", [True, True, True, True, True])
    # Começa precisando praticar/revisar e termina podendo avançar.
    assert historico[0]["acao"] in {"revisar", "praticar"}
    assert historico[-1]["acao"] == "avançar"
    assert historico[-1]["dominio"] > historico[0]["dominio"]
