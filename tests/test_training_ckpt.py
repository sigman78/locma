from locma.envs.training import _ckpt_path


def test_ckpt_path_zip_suffix():
    assert _ckpt_path("runs/ppo-greedy.zip", 1_000_000) == "runs/ppo-greedy-1000000.zip"


def test_ckpt_path_no_suffix():
    assert _ckpt_path("model", 500) == "model-500.zip"
