"""Testes mínimos do assistente de RAG.

Rodam sem nenhuma dependência externa (sem Ollama, sem ChromaDB), pois exercitam
a indexação, a recuperação, o filtro por disciplina e o tratamento da ausência.
Execute com: pytest projects/m09-rag-assistant/
"""

from rag_assistant import AssistenteRAG, BASE_EXEMPLO


def montar_assistente():
    a = AssistenteRAG()
    a.indexar(BASE_EXEMPLO)
    return a


def test_recupera_trecho_relevante():
    a = montar_assistente()
    recuperados = a.recuperar("o que é a derivada de uma função?", k=1)
    assert recuperados, "deveria recuperar ao menos um trecho"
    score, doc = recuperados[0]
    assert "derivada" in doc.texto.lower()


def test_filtro_por_disciplina():
    a = montar_assistente()
    recuperados = a.recuperar("o que é uma matriz?", disciplina="algebra", k=3)
    assert recuperados, "deveria recuperar trechos de álgebra"
    assert all(doc.disciplina == "algebra" for _, doc in recuperados)


def test_trata_ausencia():
    a = montar_assistente()
    r = a.responder("Qual a capital da Mongólia?")
    assert r.encontrou is False
    assert "não encontrei" in r.resposta.lower()
    assert r.fontes == []


def test_responder_devolve_fontes():
    a = montar_assistente()
    r = a.responder("como sei se uma matriz é invertível?", disciplina="algebra")
    assert r.encontrou is True
    assert len(r.fontes) >= 1
