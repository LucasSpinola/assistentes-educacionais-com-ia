"""Assistente Educacional Multi-Agente, projeto final da trilha (Módulo 14).

Reúne, em uma sessão de tutoria viva, as versões ricas de cada peça que a trilha
construiu:

- Calculadora segura como ferramenta (Módulo 10), com avaliador ast, sem eval.
- RAG por trecho com citação da fonte e tratamento honesto da ausência (Módulo 9).
- Roteador que decide sozinho o que fazer com a mensagem livre do aluno, o agent
  loop do Módulo 10, em vez de um switch sobre um tipo pré-rotulado.
- Banco de exercícios: o assistente propõe a questão, corrige com tolerância e
  realimenta o modelo e o analytics (Módulos 11 e 13).
- Modelo do aluno com knowledge tracing, recomendação e persistência (Módulo 13),
  mais um grafo de pré-requisitos para sugerir o próximo tema.
- Learning Analytics com engajamento, barras de domínio, risco e relatório de
  sessão (Módulo 12).

Roda do zero, só com a biblioteca padrão do Python. O Ollama é opcional: sem ele,
o Tutor devolve uma explicação extrativa a partir do trecho recuperado. Rode o
modo interativo com `python final_assistant.py` ou a sessão roteirizada com
`python final_assistant.py --demo`.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field


# ===========================================================================
# Módulo 10: calculadora segura (ferramenta)
# ===========================================================================
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


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


# ===========================================================================
# Módulo 9: RAG por trecho, com citação da fonte
# ===========================================================================
_STOPWORDS = {"a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do",
              "das", "dos", "em", "na", "no", "nas", "nos", "que", "é", "e", "ou",
              "com", "qual", "quais", "para", "por", "se", "ao", "aos", "como",
              "ela", "ele", "isso", "ser", "sua", "seu"}


def _tok(texto: str) -> list[str]:
    return [p for p in re.findall(r"\w+", texto.lower()) if p not in _STOPWORDS]


@dataclass
class Documento:
    """Um trecho de nota de aula, com o seu tema e a sua disciplina (metadados)."""

    texto: str
    tema: str
    disciplina: str


@dataclass
class Trecho:
    """Resultado de uma recuperação: o documento e o seu score de similaridade."""

    documento: Documento
    score: float


class BaseConhecimento:
    """Busca por similaridade (TF-IDF + cosseno) sobre trechos do material.

    Reusa o padrão do projeto do Módulo 9: indexa documentos com metadados,
    recupera os trechos mais relevantes e trata a ausência de forma honesta.
    """

    def __init__(self, documentos: list[Documento], limiar: float = 0.06):
        self.documentos = list(documentos)
        self.limiar = limiar
        n = len(self.documentos)
        df: Counter = Counter()
        for doc in self.documentos:
            for w in set(_tok(doc.texto)):
                df[w] += 1
        self.idf = {w: math.log(n / f) for w, f in df.items()}
        self.vetores = [self._vetor(d.texto) for d in self.documentos]

    def _vetor(self, texto: str) -> dict:
        tf = Counter(_tok(texto))
        return {w: c * self.idf.get(w, 0.0) for w, c in tf.items()}

    @staticmethod
    def _cos(a: dict, b: dict) -> float:
        prod = sum(p * b.get(w, 0.0) for w, p in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return prod / (na * nb) if na and nb else 0.0

    def buscar(self, pergunta: str, k: int = 3) -> list[Trecho]:
        """Retorna os top-k trechos acima do limiar, do mais para o menos similar."""
        q = self._vetor(pergunta)
        ranking = sorted(
            (Trecho(self.documentos[i], self._cos(q, self.vetores[i]))
             for i in range(len(self.documentos))),
            key=lambda t: t.score, reverse=True,
        )
        return [t for t in ranking[:k] if t.score >= self.limiar]


def gerar_explicacao(trecho: Documento, pergunta: str, profundidade: str,
                     modelo_llm: str = "llama3.1") -> str | None:
    """Geração opcional via Ollama (Módulos 7 e 8). Sem Ollama, devolve None."""
    prompt = (
        "Você é um tutor. Responda à pergunta do aluno usando APENAS o contexto "
        f"abaixo, em uma explicação {profundidade}. Cite a fonte ao final.\n\n"
        f"Contexto: {trecho.texto} (fonte: {trecho.tema})\n"
        f"Pergunta: {pergunta}\nResposta:"
    )
    try:
        import ollama

        r = ollama.chat(model=modelo_llm,
                        messages=[{"role": "user", "content": prompt}])
        return r["message"]["content"].strip()
    except Exception:
        return None


# ===========================================================================
# Módulo 13: modelo do aluno (knowledge tracing, recomendação, persistência)
# ===========================================================================
def atualizar_maestria(p, correto, p_aprender=0.2, p_deslize=0.1, p_chute=0.2):
    """Knowledge tracing: atualiza a probabilidade de domínio dada uma resposta."""
    if correto:
        num, den = p * (1 - p_deslize), p * (1 - p_deslize) + (1 - p) * p_chute
    else:
        num, den = p * p_deslize, p * p_deslize + (1 - p) * (1 - p_chute)
    cond = num / den if den else p
    return round(cond + (1 - cond) * p_aprender, 3)


def profundidade_por_dominio(dominio: float) -> str:
    """Mapeia o domínio para a profundidade da explicação do Tutor."""
    if dominio < 0.4:
        return "detalhada, nível iniciante"
    if dominio < 0.8:
        return "intermediária"
    return "breve, nível avançado"


# Grafo de pré-requisitos entre temas: cada tema lista os que precisam vir antes.
PRE_REQUISITOS = {
    "derivada": [],
    "integral": ["derivada"],
    "matriz": [],
    "determinante": ["matriz"],
}
TEMAS = list(PRE_REQUISITOS.keys())


class ModeloAluno:
    """Modelo persistente do aluno: domínio por tema, preferências e perfil."""

    def __init__(self, nome, preferencias=None, dominio=None, nivel="iniciante"):
        self.nome = nome
        self.nivel = nivel
        self.preferencias = preferencias or []
        self.dominio = dominio or {}

    def registrar(self, tema, correto):
        self.dominio[tema] = atualizar_maestria(self.dominio.get(tema, 0.3), correto)
        return self.dominio[tema]

    def recomendacao(self, tema):
        """Ação, profundidade e dificuldade adaptadas ao domínio atual (Módulo 13)."""
        d = self.dominio.get(tema, 0.3)
        if d < 0.4:
            acao, dif = "revisar", "fácil"
        elif d < 0.8:
            acao, dif = "praticar", "médio"
        else:
            acao, dif = "avançar", "difícil"
        prof = profundidade_por_dominio(d)
        if "exemplos visuais" in self.preferencias:
            prof += ", com um diagrama"
        return {"acao": acao, "profundidade": prof, "dificuldade": dif}

    def temas_fracos(self, limiar=0.5):
        return sorted([t for t, d in self.dominio.items() if d < limiar])

    def to_dict(self):
        return {"nome": self.nome, "nivel": self.nivel,
                "preferencias": self.preferencias, "dominio": self.dominio}

    def salvar(self, caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def carregar(cls, caminho):
        with open(caminho, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["nome"], d.get("preferencias"), d.get("dominio"),
                   d.get("nivel", "iniciante"))


def proximo_tema(modelo: ModeloAluno) -> str | None:
    """Sugere o próximo tema a estudar, respeitando os pré-requisitos.

    Um tema só entra em consideração quando todos os seus pré-requisitos já estão
    dominados (domínio alto). Entre os temas desbloqueados e ainda não dominados,
    escolhe o de menor domínio, que é onde o aluno mais precisa de atenção.
    """
    desbloqueados = [
        t for t in TEMAS
        if all(modelo.dominio.get(p, 0.0) >= 0.8 for p in PRE_REQUISITOS.get(t, []))
    ]
    pendentes = [t for t in desbloqueados if modelo.dominio.get(t, 0.3) < 0.8]
    if not pendentes:
        return None
    return min(pendentes, key=lambda t: modelo.dominio.get(t, 0.3))


# ===========================================================================
# Módulos 11 e 13: banco de exercícios e correção
# ===========================================================================
@dataclass
class Exercicio:
    tema: str
    enunciado: str
    resposta: str
    dificuldade: str


EXERCICIOS = {
    "derivada": [
        Exercicio("derivada", "Qual é a derivada de f(x) = x^2 avaliada em x = 3? Responda o número.", "6", "fácil"),
        Exercicio("derivada", "Se f(x) = 3x, quanto vale f'(x)? Responda o número.", "3", "médio"),
    ],
    "integral": [
        Exercicio("integral", "Qual é a integral de f(x) = 2x de 0 a 1? Responda o número.", "1", "médio"),
    ],
    "matriz": [
        Exercicio("matriz", "Quantos elementos tem uma matriz 2 por 3? Responda o número.", "6", "fácil"),
    ],
    "determinante": [
        Exercicio("determinante", "Uma matriz com determinante zero é invertível? Responda sim ou não.", "não", "médio"),
    ],
}


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto).strip().lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def conferir(resposta: str, esperado: str) -> bool:
    """Compara a resposta do aluno com a esperada, por número ou por texto."""
    try:
        return abs(float(str(resposta).replace(",", ".")) - float(esperado)) < 1e-6
    except (TypeError, ValueError):
        return _normalizar(resposta) == _normalizar(esperado)


def escolher_exercicio(tema: str, dificuldade: str) -> Exercicio | None:
    """Escolhe um exercício do tema, preferindo a dificuldade recomendada."""
    banco = EXERCICIOS.get(tema)
    if not banco:
        return None
    for ex in banco:
        if ex.dificuldade == dificuldade:
            return ex
    return banco[0]


# ===========================================================================
# Módulo 12: Learning Analytics da sessão
# ===========================================================================
@dataclass
class Evento:
    tema: str
    tipo: str                 # "duvida" ou "exercicio"
    correto: bool | None = None


_BLOCOS = "░▏▎▍▌▋▊▉█"


class Analytics:
    """Acompanha o engajamento e o desempenho da sessão (Módulo 12)."""

    def __init__(self):
        self.eventos: list[Evento] = []

    def registrar(self, tema, tipo, correto=None):
        self.eventos.append(Evento(tema, tipo, correto))

    def _exercicios(self):
        return [e for e in self.eventos if e.tipo == "exercicio"]

    def perguntas(self):
        return sum(1 for e in self.eventos if e.tipo == "duvida")

    def acuracia(self):
        ex = self._exercicios()
        return sum(1 for e in ex if e.correto) / len(ex) if ex else 0.0

    def sequencia_acertos(self):
        seq = 0
        for e in reversed(self._exercicios()):
            if e.correto:
                seq += 1
            else:
                break
        return seq

    def relatorio(self):
        ex = self._exercicios()
        return {"perguntas": self.perguntas(), "exercicios": len(ex),
                "acertos": sum(1 for e in ex if e.correto),
                "acuracia": round(self.acuracia(), 2),
                "sequencia": self.sequencia_acertos()}

    def risco_desengajamento(self):
        """Heurística simples de risco, em Python puro (a regressão logística do
        Módulo 12 fica como extensão). Combina acurácia e erros recentes."""
        ex = self._exercicios()
        if len(ex) < 2:
            return "indeterminado"
        erros_recentes = sum(1 for e in ex[-3:] if not e.correto)
        if self.acuracia() < 0.4 or erros_recentes >= 3:
            return "alto"
        if self.acuracia() < 0.7 or erros_recentes >= 2:
            return "médio"
        return "baixo"

    @staticmethod
    def barra(dominio: float, largura: int = 10) -> str:
        cheios = int(dominio * largura)
        resto = _BLOCOS[int((dominio * largura - cheios) * 8)]
        return "█" * cheios + (resto if cheios < largura else "") + \
            "░" * (largura - cheios - 1 if cheios < largura else 0)

    def barras_dominio(self, modelo: ModeloAluno) -> list[str]:
        return [f"{tema:14} {self.barra(d)} {d:.2f}"
                for tema, d in sorted(modelo.dominio.items())]

    def relatorio_markdown(self, modelo: ModeloAluno, sugestao: str) -> str:
        rel = self.relatorio()
        linhas = [
            f"# Relatório da sessão de {modelo.nome}",
            "",
            "## Engajamento",
            f"- Perguntas feitas: {rel['perguntas']}",
            f"- Exercícios resolvidos: {rel['exercicios']}",
            f"- Acertos: {rel['acertos']} (acurácia {rel['acuracia']:.0%})",
            f"- Sequência de acertos atual: {rel['sequencia']}",
            f"- Risco de desengajamento: {self.risco_desengajamento()}",
            "",
            "## Domínio por tema",
            "```",
            *(self.barras_dominio(modelo) or ["(ainda sem dados de domínio)"]),
            "```",
            "",
            "## Orientação do Mentor",
            sugestao,
        ]
        return "\n".join(linhas)


def sugestao_mentor(modelo: ModeloAluno, analytics: Analytics) -> str:
    """Combina temas fracos, próximo tema e desempenho em uma orientação."""
    if not analytics._exercicios():
        return "Que tal começar com um exercício? É só pedir 'exercício'."
    fracos = modelo.temas_fracos()
    partes = []
    if fracos:
        partes.append(f"Sugiro revisar, com foco em: {', '.join(fracos)}.")
    else:
        partes.append("Bom desempenho, você não tem temas fracos no momento.")
    prox = proximo_tema(modelo)
    if prox:
        partes.append(f"Próximo tema recomendado: {prox}.")
    else:
        partes.append("Você já domina os temas disponíveis, parabéns!")
    return " ".join(partes)


# ===========================================================================
# Módulo 10: roteamento da mensagem livre (o agente decide o que fazer)
# ===========================================================================
def _parece_expressao(t: str) -> bool:
    return (bool(re.fullmatch(r"[\d\s.+\-*/()]+", t))
            and any(o in t for o in "+-*/")
            and any(c.isdigit() for c in t))


def _parece_pergunta(t: str) -> bool:
    t = t.strip().lower()
    return t.endswith("?") or t.startswith(
        ("o que", "qual", "quais", "como", "quando", "por que", "porque",
         "quem", "onde", "explique", "defina"))


def _tema_no_texto(t: str) -> str | None:
    t = _normalizar(t)
    for tema in TEMAS:
        if _normalizar(tema) in t:
            return tema
    return None


def rotear(texto: str, tem_pendente: bool = False) -> tuple[str, dict]:
    """Classifica a mensagem do aluno em uma ação (o agent loop do Módulo 10)."""
    t = texto.strip().lower()
    if t in {"sair", "fim", "tchau", "exit", "quit"}:
        return "sair", {}
    if t in {"ajuda", "help", "?"}:
        return "ajuda", {}
    if any(k in t for k in ("relatório", "relatorio", "como estou",
                            "meu progresso", "como vou")):
        return "relatorio", {}
    if any(k in t for k in ("exercício", "exercicio", "praticar", "quero treinar")):
        return "exercicio", {"tema": _tema_no_texto(t)}
    if _parece_expressao(texto.strip()):
        return "calcular", {"expressao": texto.strip()}
    if tem_pendente and not _parece_pergunta(texto):
        return "responder", {"resposta": texto.strip()}
    return "duvida", {"pergunta": texto.strip()}


# ===========================================================================
# O assistente completo (coordenador integrando tudo)
# ===========================================================================
AJUDA = (
    "Posso ajudar assim: faça uma pergunta de conteúdo, peça um 'exercício', "
    "digite uma conta como 28*3/4, peça um 'relatório' do seu progresso, ou "
    "digite 'sair' para encerrar."
)


class AssistenteEducacional:
    """Coordena RAG, ferramentas, agentes, analytics e o modelo do aluno."""

    def __init__(self, documentos=None, modelo: ModeloAluno | None = None):
        if documentos is None:
            documentos = BASE
        if isinstance(documentos, dict):     # aceita {tema: texto} por conveniência
            documentos = [Documento(v, k, k) for k, v in documentos.items()]
        self.base = BaseConhecimento(documentos)
        self.modelo = modelo or ModeloAluno("aluno")
        self.analytics = Analytics()
        self.pendente: Exercicio | None = None
        self.encerrar = False

    # ----- as ações do agente -------------------------------------------
    def _tutor(self, pergunta: str) -> str:
        trechos = self.base.buscar(pergunta)
        if not trechos:
            return "[Tutor] Não encontrei isso no material disponível."
        doc = trechos[0].documento
        self.analytics.registrar(doc.tema, "duvida")
        rec = self.modelo.recomendacao(doc.tema)
        gerado = gerar_explicacao(doc, pergunta, rec["profundidade"])
        corpo = gerado or doc.texto
        return (f"[Tutor, explicação {rec['profundidade']}] {corpo} "
                f"(fonte: {doc.tema}/{doc.disciplina})")

    def _calcular(self, expressao: str) -> str:
        try:
            return f"[Calculadora] {expressao} = {calcular(expressao)}"
        except Exception as erro:
            return f"[Calculadora] não consegui calcular: {erro}"

    def _propor_exercicio(self, tema: str | None) -> str:
        tema = tema or proximo_tema(self.modelo) or "derivada"
        rec = self.modelo.recomendacao(tema)
        ex = escolher_exercicio(tema, rec["dificuldade"])
        if ex is None:
            return f"[Tutor] Ainda não tenho exercícios de {tema}."
        self.pendente = ex
        return f"[Tutor] Exercício de {ex.tema} ({ex.dificuldade}): {ex.enunciado}"

    def _avaliar(self, resposta: str) -> str:
        ex = self.pendente
        self.pendente = None
        correto = conferir(resposta, ex.resposta)
        dominio = self.modelo.registrar(ex.tema, correto)
        self.analytics.registrar(ex.tema, "exercicio", correto)
        feedback = "Correto, muito bem!" if correto else f"Ainda não, era {ex.resposta}."
        return f"[Evaluator] {feedback} (domínio de {ex.tema} agora: {dominio})"

    def _relatorio(self) -> str:
        rel = self.analytics.relatorio()
        return f"[Analytics] {rel} | [Mentor] {sugestao_mentor(self.modelo, self.analytics)}"

    # ----- ponto de entrada ---------------------------------------------
    def responder(self, texto: str) -> str:
        """Roteia a mensagem livre do aluno e devolve a resposta do assistente."""
        acao, dados = rotear(texto, tem_pendente=self.pendente is not None)
        if acao == "sair":
            self.encerrar = True
            return f"[Assistente] Até a próxima, {self.modelo.nome}!"
        if acao == "ajuda":
            return f"[Assistente] {AJUDA}"
        if acao == "calcular":
            return self._calcular(dados["expressao"])
        if acao == "exercicio":
            return self._propor_exercicio(dados["tema"])
        if acao == "responder":
            return self._avaliar(dados["resposta"])
        if acao == "relatorio":
            return self._relatorio()
        return self._tutor(dados["pergunta"])

    # ----- sessão interativa --------------------------------------------
    def sessao_interativa(self, entradas=None, saida=print,
                          caminho_modelo=None, caminho_relatorio=None) -> list[str]:
        """Conversa com o aluno em laço. `entradas` injetável torna o REPL testável.

        Sem `entradas`, lê do teclado com input(). Ao encerrar, salva o modelo do
        aluno (memória de longo prazo) e gera o relatório da sessão.
        """
        fonte = iter(entradas) if entradas is not None else None
        transcript: list[str] = []

        def diz(linha: str) -> None:
            transcript.append(linha)
            saida(linha)

        diz(f"[Assistente] Olá, {self.modelo.nome}! {AJUDA}")
        while not self.encerrar:
            if fonte is None:
                try:
                    texto = input("você> ")
                except EOFError:
                    break
            else:
                texto = next(fonte, None)
                if texto is None:
                    break
            if not texto.strip():
                continue
            diz(self.responder(texto))

        sugestao = sugestao_mentor(self.modelo, self.analytics)
        relatorio = self.analytics.relatorio_markdown(self.modelo, sugestao)
        if caminho_modelo:
            self.modelo.salvar(caminho_modelo)
        if caminho_relatorio:
            with open(caminho_relatorio, "w", encoding="utf-8") as f:
                f.write(relatorio)
        diz("\n" + relatorio)
        return transcript


# Base de conhecimento de exemplo: várias notas de aula, com tema e disciplina.
BASE = [
    Documento("A derivada mede a taxa de variação instantânea de uma função em um "
              "ponto e equivale à inclinação da reta tangente.", "derivada", "cálculo"),
    Documento("A regra da cadeia serve para derivar funções compostas, multiplicando "
              "a parte externa pela parte interna.", "derivada", "cálculo"),
    Documento("A integral definida acumula valores ao longo de um intervalo e é a "
              "operação inversa da derivação.", "integral", "cálculo"),
    Documento("Uma matriz é uma tabela de números organizada em linhas e colunas, e "
              "cada matriz pode representar uma transformação linear.", "matriz", "álgebra"),
    Documento("O determinante de uma matriz quadrada indica se ela é invertível: "
              "quando vale zero, não existe inversa.", "determinante", "álgebra"),
]

# Atalho {tema: texto} para uso rápido e compatibilidade.
MATERIAL = {d.tema: d.texto for d in BASE}


def demo() -> None:
    """Sessão roteirizada de ponta a ponta, útil para uma visão rápida e os testes."""
    aluno = ModeloAluno("Ana", preferencias=["exemplos visuais"])
    assistente = AssistenteEducacional(BASE, aluno)
    mensagens = [
        "o que é a derivada?",          # Tutor: RAG + profundidade + citação
        "28*3/4",                        # Calculadora
        "quero um exercício de derivada",
        "6",                             # acerta o exercício de derivada
        "exercício de derivada",
        "3",                             # acerta de novo, domínio sobe
        "o que é a derivada?",          # agora a explicação fica mais breve
        "exercício de matriz",
        "999",                           # erra matriz
        "relatório",                     # Analytics + Mentor + próximo tema
        "sair",
    ]
    assistente.sessao_interativa(entradas=mensagens)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--demo" in argv:
        demo()
        return
    nome = input("Seu nome: ").strip() or "aluno"
    caminho = f"modelo_{_normalizar(nome).replace(' ', '_')}.json"
    try:
        modelo = ModeloAluno.carregar(caminho)
        print(f"(bem-vindo de volta, {modelo.nome}, retomando o seu progresso)")
    except FileNotFoundError:
        modelo = ModeloAluno(nome, preferencias=["exemplos visuais"])
    AssistenteEducacional(BASE, modelo).sessao_interativa(
        caminho_modelo=caminho, caminho_relatorio="relatorio_sessao.md")


if __name__ == "__main__":
    main()
