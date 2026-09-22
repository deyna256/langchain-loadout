from dataclasses import replace

from langchain_loadout import Settings, Turn


def test_settings_are_tuned_per_product_without_touching_defaults():
    tuned = replace(Settings(), load_at=0.9, max_candidates=10)

    assert (tuned.load_at, tuned.max_candidates) == (0.9, 10)
    assert Settings().load_at == 0.9


def test_turn_carries_what_the_product_passed():
    turn = Turn(request="and for April?", context="user: spending in May")
    assert (turn.request, turn.context) == ("and for April?", "user: spending in May")
