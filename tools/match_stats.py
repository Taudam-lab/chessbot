"""Statistiques d'un match entre deux bots : score, ecart Elo et sa marge.

Sans dependance : quelques formules standard, suffisantes pour savoir si un
ecart de score entre deux versions est reel ou du bruit.

- L'ecart Elo se deduit du score moyen s par la courbe logistique :
  elo = -400 * log10(1/s - 1).
- L'intervalle a 95 % vient de l'ecart-type du resultat par partie (1, 0,5 ou
  0), converti en Elo aux deux bornes.
- La LOS (likelihood of superiority) est la probabilite que A soit vraiment
  plus fort que B. Elle ne depend que des victoires et des defaites : les
  nulles n'apportent aucune information sur qui est le meilleur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Quantile de la loi normale pour un intervalle a 95 %.
_Z_95 = 1.96


def elo_from_score(score: float) -> float:
    """Ecart Elo correspondant a un score moyen (0 a 1). Infini aux bornes."""
    if score <= 0:
        return -math.inf
    if score >= 1:
        return math.inf
    # `+ 0.0` change un -0.0 (score de 50 %) en 0.0, sinon affiche "-0".
    return -400 * math.log10(1 / score - 1) + 0.0


@dataclass(frozen=True)
class MatchStats:
    games: int
    #: Score moyen de A, de 0 a 1.
    score: float
    elo: float
    elo_low: float
    elo_high: float
    #: Probabilite que A soit plus fort que B, de 0 a 1.
    los: float


def match_stats(wins: int, losses: int, draws: int) -> MatchStats:
    """Statistiques du point de vue de A (`wins` = victoires de A)."""
    games = wins + losses + draws
    if games == 0:
        return MatchStats(0, 0.5, 0.0, -math.inf, math.inf, 0.5)

    score = (wins + 0.5 * draws) / games
    variance = (
        wins * (1 - score) ** 2 + losses * score**2 + draws * (0.5 - score) ** 2
    ) / games
    margin = _Z_95 * math.sqrt(variance / games)

    decisive = wins + losses
    los = 0.5 if decisive == 0 else 0.5 * (1 + math.erf((wins - losses) / math.sqrt(2 * decisive)))

    return MatchStats(
        games=games,
        score=score,
        elo=elo_from_score(score),
        elo_low=elo_from_score(score - margin),
        elo_high=elo_from_score(score + margin),
        los=los,
    )
