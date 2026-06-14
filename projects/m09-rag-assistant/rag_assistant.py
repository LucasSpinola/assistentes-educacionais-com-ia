"""Assistente educacional de RAG, projeto do Módulo 9.

Reúne tudo o que o módulo ensinou:
- indexação de notas de aula com metadados (disciplina),
- banco vetorial em memória (TF-IDF + cosseno), sem dependências externas,
- busca semântica com filtro por disciplina,
- montagem de contexto com citação das fontes e tratamento da ausência,
- geração da resposta com um LLM local via Ollama, com fallback extrativo.

Roda do zero, só com a biblioteca padrão do Python. O Ollama é opcional: sem ele,
o assistente devolve uma resposta extrativa baseada nos trechos recuperados.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Documento:
    """Um trecho de nota de aula com o seu metadado de disciplina."""

    texto: str
    disciplina: str


@dataclass
class Resposta:
    """Resultado de uma pergunta ao assistente."""

    pergunta: str
    resposta: str
    fontes: list[str] = field(default_factory=list)
    encontrou: bool = True


# Stopwords do português: palavras comuns e pouco informativas. Removê-las (como
# no Módulo 3) evita que preposições e artigos causem casamentos espúrios na busca.
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das",
    "dos", "em", "na", "no", "nas", "nos", "que", "é", "e", "ou", "com", "qual",
    "quais", "para", "por", "se", "ao", "aos", "como", "ela", "ele", "isso",
    "um", "ser", "sua", "seu", "the",
}


def tokenizar(texto: str) -> list[str]:
    return [p for p in re.findall(r"\w+", texto.lower()) if p not in STOPWORDS]


def cosseno(a: dict, b: dict) -> float:
    produto = sum(peso * b.get(palavra, 0.0) for palavra, peso in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return produto / (na * nb) if na and nb else 0.0


class AssistenteRAG:
    """Assistente educacional baseado em RAG, do zero."""

    def __init__(self, modelo_llm: str = "llama3.1", limiar: float = 0.05):
        self.modelo_llm = modelo_llm
        self.limiar = limiar
        self._documentos: list[Documento] = []
        self._idf: dict[str, float] = {}
        self._vetores: list[dict] = []

    # ----- indexação -----------------------------------------------------
    def indexar(self, documentos: list[Documento]) -> None:
        """Indexa os documentos, calculando o IDF e o vetor de cada um."""
        self._documentos = list(documentos)
        n = len(documentos)
        df: Counter = Counter()
        for doc in documentos:
            for palavra in set(tokenizar(doc.texto)):
                df[palavra] += 1
        self._idf = {palavra: math.log(n / freq) for palavra, freq in df.items()}
        self._vetores = [self._vetor(doc.texto) for doc in documentos]

    def _vetor(self, texto: str) -> dict:
        tf = Counter(tokenizar(texto))
        return {palavra: contagem * self._idf.get(palavra, 0.0)
                for palavra, contagem in tf.items()}

    # ----- recuperação ---------------------------------------------------
    def recuperar(self, pergunta: str, disciplina: str | None = None,
                  k: int = 3) -> list[tuple[float, Documento]]:
        """Retorna os top-k documentos mais similares, com filtro opcional."""
        consulta = self._vetor(pergunta)
        candidatos = [
            (cosseno(consulta, self._vetores[i]), self._documentos[i])
            for i in range(len(self._documentos))
            if disciplina is None or self._documentos[i].disciplina == disciplina
        ]
        candidatos.sort(key=lambda par: par[0], reverse=True)
        return [(s, d) for s, d in candidatos[:k] if s >= self.limiar]

    # ----- montagem do prompt -------------------------------------------
    @staticmethod
    def montar_prompt(pergunta: str, trechos: list[Documento]) -> str:
        contexto = "\n".join(f"[{i + 1}] {d.texto}" for i, d in enumerate(trechos))
        return (
            "Responda à pergunta usando APENAS o contexto numerado abaixo. "
            "Cite as fontes entre colchetes, como [1]. Se a resposta não estiver "
            "no contexto, diga que não encontrou.\n\n"
            f"Contexto:\n{contexto}\n\n"
            f"Pergunta: {pergunta}\nResposta:"
        )

    # ----- geração -------------------------------------------------------
    def _gerar_com_llm(self, prompt: str) -> str | None:
        try:
            import ollama

            r = ollama.chat(
                model=self.modelo_llm,
                messages=[{"role": "user", "content": prompt}],
            )
            return r["message"]["content"].strip()
        except Exception:
            return None

    def responder(self, pergunta: str, disciplina: str | None = None,
                  k: int = 3) -> Resposta:
        """Responde à pergunta, recuperando, montando o contexto e gerando."""
        recuperados = self.recuperar(pergunta, disciplina=disciplina, k=k)

        # Tratamento da ausência: nada relevante encontrado.
        if not recuperados:
            return Resposta(
                pergunta=pergunta,
                resposta="Não encontrei isso no material disponível.",
                fontes=[],
                encontrou=False,
            )

        trechos = [doc for _, doc in recuperados]
        prompt = self.montar_prompt(pergunta, trechos)

        resposta_llm = self._gerar_com_llm(prompt)
        if resposta_llm is None:
            # Fallback extrativo: devolve o trecho mais relevante.
            resposta_llm = (
                "(LLM indisponível, resposta extrativa) "
                f"{trechos[0].texto} [1]"
            )

        return Resposta(
            pergunta=pergunta,
            resposta=resposta_llm,
            fontes=[d.texto for d in trechos],
            encontrou=True,
        )


# Base de conhecimento de exemplo: notas de aula de duas disciplinas.
BASE_EXEMPLO = [
    Documento("A derivada mede a taxa de variação instantânea de uma função "
              "em um ponto, e equivale à inclinação da reta tangente.", "calculo"),
    Documento("A regra da cadeia permite derivar funções compostas, "
              "multiplicando a derivada externa pela interna.", "calculo"),
    Documento("A integral definida acumula valores ao longo de um intervalo "
              "e é a operação inversa da derivada.", "calculo"),
    Documento("Uma matriz é uma tabela de números organizada em linhas e "
              "colunas, usada para representar transformações lineares.", "algebra"),
    Documento("O determinante de uma matriz quadrada indica se ela é "
              "invertível: se for zero, a matriz não tem inversa.", "algebra"),
    Documento("Um autovetor de uma matriz é um vetor cuja direção não muda "
              "quando a matriz é aplicada, apenas a sua escala.", "algebra"),
]


def demo() -> None:
    """Demonstração de ponta a ponta do assistente."""
    assistente = AssistenteRAG()
    assistente.indexar(BASE_EXEMPLO)

    perguntas = [
        ("O que é a derivada de uma função?", None),
        ("Como sei se uma matriz é invertível?", "algebra"),
        ("Qual a capital da Mongólia?", None),   # fora do material
    ]
    for pergunta, disciplina in perguntas:
        r = assistente.responder(pergunta, disciplina=disciplina)
        print("Pergunta:", pergunta, f"(disciplina={disciplina})")
        print("  Resposta:", r.resposta)
        print("  Encontrou:", r.encontrou, "| Fontes:", len(r.fontes))
        print()


if __name__ == "__main__":
    demo()
