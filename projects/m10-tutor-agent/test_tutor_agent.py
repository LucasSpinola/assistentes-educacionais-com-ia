"""Testes mínimos do agente tutor.

Rodam sem dependências externas (sem Ollama), pois exercitam o roteamento de
ferramentas, a calculadora, a busca, a memória e o planejamento de etapas.
Execute com: pytest projects/m10-tutor-agent/
"""

from tutor_agent import AgenteTutor, BaseConhecimento, DOCUMENTOS_EXEMPLO, calcular


def montar_tutor():
    return AgenteTutor(BaseConhecimento(DOCUMENTOS_EXEMPLO))


def test_calculadora_segura():
    assert calcular("28*3/4") == 21.0
    assert calcular("2**10") == 1024


def test_bloqueia_codigo_malicioso():
    bloqueou = False
    try:
        calcular("__import__('os')")
    except Exception:
        bloqueou = True
    assert bloqueou, "a calculadora deveria bloquear código não aritmético"


def test_roteia_conta_para_calculadora():
    tutor = montar_tutor()
    acao = tutor.responder("quanto é 28*3/4 ?")
    assert acao.ferramenta == "calcular"
    assert acao.observacao == 21.0


def test_roteia_conteudo_para_busca():
    tutor = montar_tutor()
    acao = tutor.responder("o que é a derivada?")
    assert acao.ferramenta == "buscar"
    assert "derivada" in acao.observacao.lower()


def test_memoria_personaliza_resposta():
    tutor = montar_tutor()
    tutor.memoria.lembrar("nome", "Ana")
    acao = tutor.responder("o que é a derivada?")
    assert acao.resposta.startswith("Ana,")


def test_busca_trata_ausencia():
    tutor = montar_tutor()
    acao = tutor.responder("qual a capital da França?")
    assert "não encontrei" in acao.observacao.lower()


def test_planejamento_encadeia_passos():
    tutor = montar_tutor()
    acoes = tutor.resolver_problema([("Cadernos", "28*3"), ("Pacotes", "{r}/4")])
    assert acoes[0].observacao == 84
    assert acoes[-1].observacao == 21.0
