"""Registre des adversaires disponibles.

Ajouter un bot = l'instancier ici : l'arene (`tools/arena.py`) le retrouve par
son nom a partir de cette liste.
"""

from app.bots.base import Bot
from app.bots.minimax import MinimaxBot
from app.bots.random_bot import RandomBot

_REGISTRY: dict[str, Bot] = {
    bot.name: bot
    for bot in (
        RandomBot(),
        MinimaxBot(
            name="easy",
            label="Minimax - Facile",
            max_depth=2,
            time_budget=0.20,
            # Sans quiescence il ne voit pas le bout des echanges : il gaffe
            # comme un debutant au lieu de jouer juste mais court.
            use_quiescence=False,
        ),
        MinimaxBot(name="normal", label="Minimax - Normal", time_budget=0.60),
        MinimaxBot(name="hard", label="Minimax - Difficile", time_budget=1.50),
        # Le moteur deploye sur Lichess (app/uci.py le recupere ici) : l'arene
        # teste ainsi exactement la configuration en production. Plafond
        # large, c'est la pendule qui limite : a jouer avec `--tc` dans
        # l'arene, sinon il reflechit 120 s par coup.
        MinimaxBot(name="lichess", label="Minimax - Lichess", time_budget=120.0),
    )
}

DEFAULT_BOT = "normal"


def get_bot(name: str | None) -> Bot | None:
    """Renvoie le bot demande, ou None si le nom est inconnu."""
    return _REGISTRY.get(name or DEFAULT_BOT)


def list_bots() -> list[dict[str, str]]:
    return [{"name": bot.name, "label": bot.label} for bot in _REGISTRY.values()]


__all__ = ["Bot", "DEFAULT_BOT", "get_bot", "list_bots"]
