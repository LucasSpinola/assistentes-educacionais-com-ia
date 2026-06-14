"""Dashboard de aprendizado, projeto do Módulo 12.

Reúne tudo o que o módulo ensinou:
- coleta de dados de uma turma (features de cada aluno),
- métricas e um índice de engajamento (frequência, volume, constância),
- um classificador de evasão (regressão logística, do Módulo 2),
- um painel que mostra o engajamento e o risco de evasão de cada aluno,
  destacando os que precisam de atenção.

Roda do zero, só com numpy. O Streamlit é opcional para uma versão em app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def score_engajamento(dias_ativos, total_dias, exercicios, sequencia) -> float:
    """Combina frequência, volume e constância em um índice de 0 a 1."""
    frequencia = dias_ativos / total_dias if total_dias else 0.0
    volume = min(exercicios / 20, 1.0)
    constancia = min(sequencia / 7, 1.0)
    return round(0.4 * frequencia + 0.4 * volume + 0.2 * constancia, 3)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


@dataclass
class LinhaPainel:
    aluno: str
    engajamento: float
    prob_evasao: float
    em_risco: bool


class TurmaAnalytics:
    """Gera dados de uma turma, calcula engajamento e treina o modelo de evasão."""

    def __init__(self, n: int = 200, total_dias: int = 14, seed: int = 0):
        self.total_dias = total_dias
        rng = np.random.default_rng(seed)
        self.n = n
        self.nomes = [f"aluno_{i:03d}" for i in range(n)]

        # Features de cada aluno.
        self.dias_ativos = rng.integers(0, total_dias + 1, n)
        self.exercicios = rng.integers(0, 30, n)
        self.sequencia = rng.integers(0, 10, n)
        self.acuracia = rng.uniform(0, 1, n)

        # Índice de engajamento por aluno.
        self.eng = np.array([
            score_engajamento(self.dias_ativos[i], total_dias, self.exercicios[i], self.sequencia[i])
            for i in range(n)
        ])

        # Desfecho histórico: baixo engajamento e baixa acurácia -> mais risco.
        risco = -3 * self.eng - 1.5 * self.acuracia + 2 + rng.normal(0, 0.3, n)
        self.evadiu = (risco > 0).astype(int)

        self._w = None
        self._b = 0.0
        self._media = None
        self._desvio = None

    def _features(self) -> np.ndarray:
        freq = self.dias_ativos / self.total_dias
        vol = np.minimum(self.exercicios / 20, 1.0)
        return np.column_stack([freq, vol, self.eng, self.acuracia])

    def treinar(self, iteracoes: int = 3000, taxa: float = 0.1) -> None:
        X = self._features()
        self._media, self._desvio = X.mean(0), X.std(0)
        Xn = (X - self._media) / self._desvio
        w = np.zeros(Xn.shape[1])
        b = 0.0
        for _ in range(iteracoes):
            erro = sigmoide(Xn @ w + b) - self.evadiu
            w -= taxa * Xn.T @ erro / self.n
            b -= taxa * erro.mean()
        self._w, self._b = w, b

    def prob_evasao(self) -> np.ndarray:
        if self._w is None:
            self.treinar()
        Xn = (self._features() - self._media) / self._desvio
        return sigmoide(Xn @ self._w + self._b)

    def acuracia_modelo(self) -> float:
        return float(((self.prob_evasao() >= 0.5).astype(int) == self.evadiu).mean())

    def dashboard(self, limiar: float = 0.7) -> list[LinhaPainel]:
        """Painel de cada aluno, ordenado do maior para o menor risco."""
        probs = self.prob_evasao()
        linhas = [
            LinhaPainel(self.nomes[i], float(self.eng[i]), float(probs[i]), bool(probs[i] >= limiar))
            for i in range(self.n)
        ]
        return sorted(linhas, key=lambda l: l.prob_evasao, reverse=True)

    def resumo(self, limiar: float = 0.7) -> dict:
        probs = self.prob_evasao()
        return {
            "alunos": self.n,
            "evadiram_historico": int(self.evadiu.sum()),
            "acuracia_modelo": round(self.acuracia_modelo(), 3),
            "em_risco": int((probs >= limiar).sum()),
        }


def demo() -> None:
    turma = TurmaAnalytics(n=200, seed=0)
    turma.treinar()

    print("=== Resumo da turma ===")
    for chave, valor in turma.resumo().items():
        print(f"  {chave}: {valor}")

    print("\n=== Top 8 alunos em risco (priorizar atenção) ===")
    print(f"{'aluno':12} {'engajamento':>12} {'risco evasão':>14}")
    for linha in turma.dashboard()[:8]:
        flag = "  <- em risco" if linha.em_risco else ""
        print(f"{linha.aluno:12} {linha.engajamento:>12.3f} {linha.prob_evasao:>13.1%}{flag}")


if __name__ == "__main__":
    demo()
