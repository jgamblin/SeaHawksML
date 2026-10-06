"""Regime-change flag: did the team change head coach since last season's final game?"""

import polars as pl


def new_head_coach(team_games: pl.DataFrame) -> pl.DataFrame:
    """Per (game_id, team): 1 if the team's first coach this season differs from the coach of
    its final game last season. Constant within a season (mid-season changes not flagged)."""
    first: dict[tuple[str, int], str | None] = {}
    last: dict[tuple[str, int], str | None] = {}
    for r in team_games.sort("kickoff_utc", "game_id").iter_rows(named=True):
        key = (r["team"], r["season"])
        first.setdefault(key, r["coach"])
        last[key] = r["coach"]
    rows = []
    for r in team_games.iter_rows(named=True):
        prev = last.get((r["team"], r["season"] - 1))
        cur = first[(r["team"], r["season"])]
        flag = int(prev is not None and cur is not None and cur != prev)
        rows.append({"game_id": r["game_id"], "team": r["team"], "new_head_coach": flag})
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "team": pl.Utf8, "new_head_coach": pl.Int64})
