"""Sistema adaptativo de aprendizagem, projeto do Módulo 13.

Reúne tudo o que o módulo ensinou:
- um modelo do aluno (perfil) com domínio por habilidade,
- knowledge tracing para estimar o domínio a cada resposta,
- persistência do modelo entre sessões (JSON),
- um recomendador que adapta ação, profundidade e dificuldade ao domínio.

Roda do zero, só com a biblioteca padrão do Python.
"""

from __future__ import annotations

import json


def atualizar_maestria(p_dominio, correto, p_aprender=0.2, p_deslize=0.1, p_chute=0.2):
    """Knowledge tracing: atualiza a probabilidade de domínio dada uma resposta."""
    if correto:
        num = p_dominio * (1 - p_deslize)
        den = p_dominio * (1 - p_deslize) + (1 - p_dominio) * p_chute
    else:
        num = p_dominio * p_deslize
        den = p_dominio * p_deslize + (1 - p_dominio) * (1 - p_chute)
    p_cond = num / den if den else p_dominio
    return round(p_cond + (1 - p_cond) * p_aprender, 3)


def recomendar(dominio, preferencias=None):
    """Decide a ação, a profundidade e a dificuldade a partir do domínio."""
    preferencias = preferencias or []
    if dominio < 0.4:
        acao, profundidade, dificuldade = "revisar", "explicação detalhada", "exercício fácil"
    elif dominio < 0.8:
        acao, profundidade, dificuldade = "praticar", "explicação intermediária", "exercício médio"
    else:
        acao, profundidade, dificuldade = "avançar", "explicação breve", "exercício difícil"
    if "exemplos visuais" in preferencias:
        profundidade += ", com um diagrama"
    return {"acao": acao, "profundidade": profundidade, "dificuldade": dificuldade}


class ModeloAluno:
    """Modelo persistente do aluno: perfil + domínio por habilidade (knowledge tracing)."""

    def __init__(self, nome, nivel="iniciante", preferencias=None, dominio=None):
        self.nome = nome
        self.nivel = nivel
        self.preferencias = preferencias or []
        self.dominio = dominio or {}          # habilidade -> probabilidade de domínio

    def registrar_resposta(self, habilidade, correto):
        """Atualiza o domínio da habilidade com knowledge tracing."""
        atual = self.dominio.get(habilidade, 0.3)
        self.dominio[habilidade] = atualizar_maestria(atual, correto)
        return self.dominio[habilidade]

    def recomendacao(self, habilidade):
        """Recomendação adaptativa para a habilidade, com base no domínio atual."""
        return recomendar(self.dominio.get(habilidade, 0.3), self.preferencias)

    # ----- persistência --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "nome": self.nome,
            "nivel": self.nivel,
            "preferencias": self.preferencias,
            "dominio": self.dominio,
        }

    def salvar(self, caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def carregar(cls, caminho):
        with open(caminho, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["nome"], d["nivel"], d["preferencias"], d["dominio"])


def simular_trajetoria(modelo: ModeloAluno, habilidade: str, respostas: list[bool]) -> list[dict]:
    """Processa uma sequência de respostas e devolve a recomendação a cada passo."""
    historico = []
    for correto in respostas:
        dominio = modelo.registrar_resposta(habilidade, correto)
        rec = modelo.recomendacao(habilidade)
        historico.append({"correto": correto, "dominio": dominio, "acao": rec["acao"]})
    return historico


def demo() -> None:
    aluno = ModeloAluno("Ana", nivel="iniciante", preferencias=["exemplos visuais"])

    print("=== Trajetória da Ana em 'derivada' (acertando cada vez mais) ===")
    respostas = [False, True, True, True, True, True]
    for passo in simular_trajetoria(aluno, "derivada", respostas):
        marca = "acertou" if passo["correto"] else "errou  "
        print(f"  {marca} | domínio={passo['dominio']:.3f} | recomendação: {passo['acao']}")

    print("\n=== Recomendação final ===")
    rec = aluno.recomendacao("derivada")
    print(f"  {rec['acao']}: {rec['profundidade']}, {rec['dificuldade']}")


if __name__ == "__main__":
    demo()
