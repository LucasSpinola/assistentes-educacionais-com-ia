"""Testes de integração do assistente educacional final.

Rodam sem dependências externas, exercitando todas as peças integradas: o
roteamento do agente, o RAG com citação, a calculadora, o banco de exercícios, o
modelo do aluno com knowledge tracing e pré-requisitos, o analytics e a sessão
interativa com persistência. Execute com: pytest projects/m14-final-assistant/
"""

import os
import tempfile

from final_assistant import (
    AssistenteEducacional, ModeloAluno, calcular, conferir, rotear,
    proximo_tema, EXERCICIOS,
)


def montar():
    return AssistenteEducacional(modelo=ModeloAluno("Ana", preferencias=["exemplos visuais"]))


# ----- ferramenta de cálculo (M10) ---------------------------------------
def test_calculadora_funciona():
    assert "21.0" in montar().responder("28*3/4")


def test_calculadora_bloqueia_codigo():
    bloqueou = False
    try:
        calcular("__import__('os')")
    except Exception:
        bloqueou = True
    assert bloqueou


# ----- roteamento do agente (M10) ----------------------------------------
def test_roteamento_decide_a_acao():
    assert rotear("28*3/4")[0] == "calcular"
    assert rotear("quero um exercício de derivada")[0] == "exercicio"
    assert rotear("como estou indo?")[0] == "relatorio"
    assert rotear("o que é a derivada?")[0] == "duvida"
    assert rotear("sair")[0] == "sair"
    # Com exercício pendente, uma resposta curta vira avaliação, não dúvida.
    assert rotear("6", tem_pendente=True)[0] == "responder"


# ----- Tutor com RAG e citação (M9) --------------------------------------
def test_tutor_busca_e_cita_a_fonte():
    r = montar().responder("o que é a derivada?")
    assert "Tutor" in r and "derivada" in r.lower() and "fonte:" in r


def test_tutor_trata_pergunta_fora_do_material():
    assert "não encontrei" in montar().responder("qual a capital da França?").lower()


# ----- banco de exercícios e correção (M11, M13) -------------------------
def test_exercicio_proposto_e_corrigido_atualiza_modelo_e_analytics():
    a = montar()
    proposta = a.responder("exercício de derivada")
    assert "Exercício de derivada" in proposta and a.pendente is not None
    a.responder(a.pendente.resposta)               # responde com o gabarito
    assert a.modelo.dominio["derivada"] > 0.3      # acerto eleva o domínio
    assert a.analytics.relatorio()["exercicios"] == 1
    assert a.analytics.relatorio()["acertos"] == 1


def test_exercicio_errado_baixa_o_dominio():
    a = montar()
    a.responder("exercício de matriz")
    a.responder("resposta claramente errada 9999")
    assert a.modelo.dominio["matriz"] < 0.3
    assert "matriz" in a.modelo.temas_fracos()


def test_conferir_numero_e_texto():
    assert conferir("6", "6") and conferir("6.0", "6")
    assert conferir("Não", "não")                  # ignora acento e caixa
    assert not conferir("7", "6")


# ----- modelo do aluno: pré-requisitos e personalização (M13) ------------
def test_proximo_tema_respeita_prerequisitos():
    aluno = ModeloAluno("Ana")
    # Sem dominar matriz, o determinante (que depende dela) não é recomendado.
    assert proximo_tema(aluno) in ("derivada", "matriz")
    aluno.dominio["matriz"] = 0.9                  # destrava o determinante
    aluno.dominio["derivada"] = 0.9
    aluno.dominio["integral"] = 0.9
    assert proximo_tema(aluno) == "determinante"


def test_personalizacao_adapta_profundidade():
    a = montar()
    assert "detalhada" in a.responder("o que é a derivada?")
    for _ in range(4):                             # vários acertos elevam o domínio
        a.modelo.registrar("derivada", True)
    assert "breve" in a.responder("o que é a derivada?")


# ----- analytics e mentor (M12, M11) -------------------------------------
def test_relatorio_traz_metricas_risco_e_sugestao():
    a = montar()
    a.responder("exercício de matriz")
    a.responder("9999")                            # erra
    r = a.responder("relatório")
    assert "Analytics" in r and "Mentor" in r
    assert "revisar" in r.lower() and "matriz" in r.lower()
    rel = a.analytics.relatorio()
    assert set(rel) == {"perguntas", "exercicios", "acertos", "acuracia", "sequencia"}
    assert a.analytics.risco_desengajamento() in {"indeterminado", "baixo", "médio", "alto"}


# ----- sessão interativa e persistência (M13) ----------------------------
def test_sessao_interativa_injetavel_e_persistencia():
    caminho = os.path.join(tempfile.gettempdir(), "modelo_teste_m14.json")
    if os.path.exists(caminho):
        os.remove(caminho)
    a = montar()
    transcript = a.sessao_interativa(
        entradas=["o que é a derivada?", "exercício de derivada", "6", "relatório", "sair"],
        saida=lambda _l: None, caminho_modelo=caminho,
    )
    texto = "\n".join(transcript)
    assert "fonte:" in texto                       # o Tutor citou a fonte
    assert "Correto" in texto                       # o exercício foi corrigido
    assert "Relatório da sessão" in texto           # o relatório foi gerado
    # A memória de longo prazo: o modelo salvo recarrega com o domínio evoluído.
    recarregado = ModeloAluno.carregar(caminho)
    assert recarregado.dominio["derivada"] > 0.3
    os.remove(caminho)


def test_persistencia_ida_e_volta():
    caminho = os.path.join(tempfile.gettempdir(), "modelo_rt_m14.json")
    aluno = ModeloAluno("Bia", preferencias=["exemplos visuais"], dominio={"derivada": 0.71})
    aluno.salvar(caminho)
    volta = ModeloAluno.carregar(caminho)
    assert volta.nome == "Bia" and volta.preferencias == ["exemplos visuais"]
    assert volta.dominio == {"derivada": 0.71}
    os.remove(caminho)
