import pytest

from locma.envs.training import ZOO_OPPONENTS, train_zoo
from locma.policies.registry import make_policy


def test_zoo_opponents_are_valid_specs():
    assert isinstance(ZOO_OPPONENTS, tuple) and len(ZOO_OPPONENTS) >= 1
    for spec in ZOO_OPPONENTS:
        make_policy(spec)  # each declared opponent must be a buildable policy spec


def test_train_zoo_rejects_empty_opponents():
    # The empty-list guard fires before the lazy MaskablePPO import, so this is
    # testable without the [ml] extra.
    with pytest.raises(ValueError, match="non-empty"):
        train_zoo(opponents=[], out="unused.zip")
