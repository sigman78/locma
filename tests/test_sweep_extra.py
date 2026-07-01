import tomllib
from pathlib import Path


def test_sweep_extra_declares_optuna_and_tensorboard():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    sweep = data["project"]["optional-dependencies"]["sweep"]
    joined = " ".join(sweep)
    assert "optuna" in joined
    assert "tensorboard" in joined
