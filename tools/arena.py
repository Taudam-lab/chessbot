"""Arene bot-vs-bot : fait jouer deux bots du registre l'un contre l'autre et
affiche un score.

Outil de developpement : les bots sont appeles directement en process, sans
passer par UCI ni par le reseau.

Options :
- `--tc 60+1` : une pendule par camp (base en secondes + increment par coup).
  Chaque bot recoit son temps restant comme sur Lichess, son temps reel de
  reflexion est decompte, et une pendule a zero perd la partie. Sans `--tc`,
  chaque bot pense a son budget fixe (`time_budget`, declare dans
  `app/bots/__init__.py`) : la gestion du temps n'est alors pas testee.
- `--openings` : chaque paire de parties part d'une ligne de theorie de
  `tools/build_book.py`, couleurs inversees entre les deux parties de la paire.
  Sans cela toutes les parties partent de la position initiale et se
  ressemblent beaucoup.
- `--workers N` : parties en parallele, une par process (la recherche est
  CPU-bound, des fils seraient bloques par le GIL).

Le bilan donne le score, l'ecart Elo avec sa marge a 95 % et la probabilite
que A soit plus fort que B (voir `tools/match_stats.py`).

Pistes non traitees : mode a profondeur ou noeuds fixes, pour des resultats
reproductibles d'un run a l'autre, et arret sequentiel (SPRT).

Usage (depuis la racine du projet) :
    python -m tools.arena --bot-a random --bot-b normal --games 10
    python -m tools.arena --bot-a normal --bot-b lichess --tc 10+0.1 --openings --games 40 --workers 6
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import chess

from app.bots import get_bot
from tools.build_book import OPENING_LINES
from tools.match_stats import match_stats

DEFAULT_GAMES = 10
#: Garde-fou contre une partie qui ne se termine jamais (bug de repetition,
#: bot qui refuse de mater). Une partie humaine depasse rarement 150 coups.
DEFAULT_MAX_PLIES = 300

#: Au-dela, un bot sans pendule reflechit si longtemps par coup que le match
#: devient interminable : c'est le signe qu'il attend une pendule (`--tc`).
_LONG_BUDGET_WARNING = 10.0
#: Coeurs laisses libres quand la pendule tourne : des parties qui se disputent
#: le processeur jouent moins vite que prevu et faussent la gestion du temps.
_FREE_CORES_WITH_CLOCK = 2


@dataclass(frozen=True)
class TimeControl:
    base_ms: int
    increment_ms: int

    @classmethod
    def parse(cls, text: str) -> TimeControl:
        """`60+1` : 60 s plus 1 s par coup. `60` : sans increment."""
        base, _, increment = text.partition("+")
        try:
            return cls(round(float(base) * 1000), round(float(increment or 0) * 1000))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"cadence invalide : {text!r} (attendu par exemple : 60+1)"
            ) from None

    def __str__(self) -> str:
        return f"{self.base_ms / 1000:g}+{self.increment_ms / 1000:g}"


@dataclass(frozen=True)
class GameResult:
    #: `chess.WHITE`, `chess.BLACK`, ou None pour une nulle.
    winner: chess.Color | None
    #: Vrai si une pendule est tombee a zero.
    on_time: bool
    #: Temps restant (Blancs, Noirs) en fin de partie, en ms ; None sans pendule.
    clocks: tuple[int, int] | None


@dataclass(frozen=True)
class _GameSpec:
    index: int
    a_plays_white: bool
    opening_name: str | None
    #: Coups de l'ouverture, en SAN, joues avant que les bots prennent la main.
    opening: tuple[str, ...]


@dataclass
class _Score:
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    time_losses_a: int = 0
    time_losses_b: int = 0

    def record(self, winner: str | None, on_time: bool = False) -> None:
        if winner == "a":
            self.wins_a += 1
            self.time_losses_b += on_time
        elif winner == "b":
            self.wins_b += 1
            self.time_losses_a += on_time
        else:
            # Une pendule tombee face a un camp qui ne peut plus mater donne
            # la nulle : ce n'est pas compte comme une defaite au temps.
            self.draws += 1


def _play_game(
    bot_white,
    bot_black,
    max_plies: int,
    time_control: TimeControl | None = None,
    opening: tuple[str, ...] = (),
) -> GameResult:
    """Joue une partie entre deux bots.

    `claim_draw=True` fait compter les nulles reclamables (repetition, regle
    des 50 coups) : sans ca, une partie moteur-contre-moteur peut boucler bien
    au-dela de `max_plies` avant qu'une nulle "automatique" (mat, pat, materiel
    insuffisant) ne survienne.
    """
    board = chess.Board()
    for san in opening:
        board.push_san(san)

    clocks: dict[chess.Color, int] | None = None
    if time_control is not None:
        clocks = {chess.WHITE: time_control.base_ms, chess.BLACK: time_control.base_ms}

    def clock_pair() -> tuple[int, int] | None:
        return None if clocks is None else (clocks[chess.WHITE], clocks[chess.BLACK])

    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        color = board.turn
        bot = bot_white if color == chess.WHITE else bot_black
        if clocks is None:
            move = bot.choose_move(board)
        else:
            started = time.monotonic()
            move = bot.choose_move(
                board, ms_left=clocks[color], increment_ms=time_control.increment_ms
            )
            clocks[color] -= round((time.monotonic() - started) * 1000)
            if clocks[color] < 0:
                # Comme sur Lichess : perdre au temps face a un camp qui n'a
                # plus de quoi mater donne la nulle.
                opponent = not color
                winner = None if board.has_insufficient_material(opponent) else opponent
                return GameResult(winner, on_time=True, clocks=clock_pair())
            # L'increment est rendu apres le coup, comme sur une vraie pendule.
            clocks[color] += time_control.increment_ms
        if move is None:
            break
        board.push(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None else outcome.winner
    return GameResult(winner, on_time=False, clocks=clock_pair())


def _game_specs(games: int, openings: bool) -> list[_GameSpec]:
    """Parties appariees : A joue Blancs sur les parties paires, Noirs sur les
    impaires, et les deux parties d'une paire partent de la meme ouverture.
    Sans cela, l'avantage du trait et le choix d'ouverture fausseraient le
    score."""
    specs = []
    for index in range(games):
        if openings:
            name, moves, _weight = OPENING_LINES[(index // 2) % len(OPENING_LINES)]
            specs.append(_GameSpec(index, index % 2 == 0, name, tuple(moves)))
        else:
            specs.append(_GameSpec(index, index % 2 == 0, None, ()))
    return specs


def _play_spec(bot_a, bot_b, spec: _GameSpec, max_plies: int, time_control) -> GameResult:
    white, black = (bot_a, bot_b) if spec.a_plays_white else (bot_b, bot_a)
    return _play_game(white, black, max_plies, time_control, spec.opening)


def _play_spec_by_name(
    name_a: str, name_b: str, spec: _GameSpec, max_plies: int, time_control
) -> tuple[_GameSpec, GameResult]:
    """Point d'entree d'un worker. Les bots sont retrouves par leur nom : seuls
    des noms et des resultats traversent la frontiere entre process."""
    return spec, _play_spec(get_bot(name_a), get_bot(name_b), spec, max_plies, time_control)


def _record_and_print(spec: _GameSpec, result: GameResult, games: int, score: _Score) -> None:
    if result.winner is None:
        winner = None
    elif (result.winner == chess.WHITE) == spec.a_plays_white:
        winner = "a"
    else:
        winner = "b"
    score.record(winner, result.on_time)

    outcome = {"a": "A", "b": "B", None: "nulle"}[winner]
    if result.on_time:
        outcome += " (au temps)"
    line = f"Partie {spec.index + 1}/{games} : {'A' if spec.a_plays_white else 'B'} a les Blancs"
    if spec.opening_name:
        line += f", {spec.opening_name}"
    line += f" -> {outcome}"
    if result.clocks is not None:
        white_ms, black_ms = result.clocks
        line += f"  [pendules : Blancs {white_ms / 1000:.1f}s, Noirs {black_ms / 1000:.1f}s]"
    print(line, flush=True)


def run_match_with_bots(
    bot_a,
    bot_b,
    games: int,
    max_plies: int,
    time_control: TimeControl | None = None,
    openings: bool = False,
) -> _Score:
    """Comme `run_match`, mais avec des bots deja instancies, en sequence :
    permet de tester un bot hors registre (ex. `tools/arena_nnue.py`) sans
    passer par `get_bot`.
    """
    score = _Score()
    for spec in _game_specs(games, openings):
        result = _play_spec(bot_a, bot_b, spec, max_plies, time_control)
        _record_and_print(spec, result, games, score)
    return score


def run_match(
    name_a: str,
    name_b: str,
    games: int,
    max_plies: int,
    time_control: TimeControl | None = None,
    openings: bool = False,
    workers: int = 1,
) -> _Score:
    bot_a = get_bot(name_a)
    bot_b = get_bot(name_b)
    if bot_a is None:
        raise SystemExit(f"Bot inconnu : {name_a}")
    if bot_b is None:
        raise SystemExit(f"Bot inconnu : {name_b}")

    if time_control is None:
        for name, bot in ((name_a, bot_a), (name_b, bot_b)):
            budget = getattr(bot, "time_budget", 0.0)
            if budget > _LONG_BUDGET_WARNING:
                print(
                    f"Attention : sans --tc, {name} reflechit jusqu'a {budget:g} s par coup "
                    "(il attend une pendule)."
                )

    if workers <= 1:
        return run_match_with_bots(bot_a, bot_b, games, max_plies, time_control, openings)

    score = _Score()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_play_spec_by_name, name_a, name_b, spec, max_plies, time_control)
            for spec in _game_specs(games, openings)
        ]
        for future in as_completed(futures):
            spec, result = future.result()
            _record_and_print(spec, result, games, score)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Fait jouer deux bots du registre l'un contre l'autre.")
    parser.add_argument("--bot-a", required=True, help="Nom du premier bot (voir app/bots/__init__.py)")
    parser.add_argument("--bot-b", required=True, help="Nom du second bot")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="Nombre de parties (defaut : %(default)s)")
    parser.add_argument(
        "--max-plies", type=int, default=DEFAULT_MAX_PLIES, help="Limite de demi-coups par partie (defaut : %(default)s)"
    )
    parser.add_argument(
        "--tc", type=TimeControl.parse, default=None, help="Pendule par camp, en secondes : base+increment (ex. 60+1)"
    )
    parser.add_argument(
        "--openings", action="store_true", help="Faire partir chaque paire de parties d'une ligne de theorie"
    )
    parser.add_argument("--workers", type=int, default=1, help="Parties jouees en parallele (defaut : %(default)s)")
    args = parser.parse_args()

    workers = args.workers
    if args.tc is not None:
        max_workers = max(1, (os.cpu_count() or 1) - _FREE_CORES_WITH_CLOCK)
        if workers > max_workers:
            print(
                f"--workers ramene a {max_workers} : avec une pendule, des parties qui se "
                "disputent le processeur fausseraient la gestion du temps."
            )
            workers = max_workers

    cadence = f"pendule {args.tc}" if args.tc is not None else "budget fixe par coup"
    print(
        f"{args.bot_a} (A) contre {args.bot_b} (B) : {args.games} parties, {cadence}, "
        f"{'ouvertures du livre' if args.openings else 'position initiale'}, {workers} worker(s)"
    )
    score = run_match(
        args.bot_a, args.bot_b, args.games, args.max_plies, args.tc, args.openings, workers
    )

    total = score.wins_a + score.wins_b + score.draws
    stats = match_stats(score.wins_a, score.wins_b, score.draws)
    print()
    print(f"Score sur {total} parties :")
    print(f"  {args.bot_a} (A) : {score.wins_a}")
    print(f"  {args.bot_b} (B) : {score.wins_b}")
    print(f"  Nulles        : {score.draws}")
    if args.tc is not None:
        print(f"  Defaites au temps : A {score.time_losses_a}, B {score.time_losses_b}")
    print()
    print(f"Score de A : {stats.score:.1%}")
    if 0 < stats.score < 1:
        print(f"Ecart Elo (A - B) : {stats.elo:+.0f}  [95 % : {stats.elo_low:+.0f} a {stats.elo_high:+.0f}]")
    else:
        # A 0 % ou 100 %, la courbe logistique part a l'infini : l'ecart est
        # seulement "tres grand", aucun chiffre n'a de sens.
        print("Ecart Elo (A - B) : non estimable, un bot a remporte toutes les parties")
    print(f"Probabilite que A soit plus fort : {stats.los:.1%}")


if __name__ == "__main__":
    main()
