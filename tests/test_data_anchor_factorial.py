from scripts.preflight_data_anchor_objectives import CONDITIONS


def test_full_nonempty_three_head_factorial_is_registered() -> None:
    observed = {
        name: (weights["reconstruction"], weights["tcremp"], weights["pgen"])
        for name, weights in CONDITIONS.items()
    }
    assert observed == {
        "r": (1.0, 0.0, 0.0),
        "t": (0.0, 1.0, 0.0),
        "p": (0.0, 0.0, 1.0),
        "rt": (1.0, 1.0, 0.0),
        "rp": (1.0, 0.0, 1.0),
        "tp": (0.0, 1.0, 1.0),
    }
