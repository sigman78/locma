from locma.stats.openskill_ratings import openskill_from_results, ordinal


def test_dominant_player_outranks_weak_one():
    # 'strong' beats 'weak' 20 times (score_a = 1.0 means a won)
    pairs = [("strong", "weak", 1.0) for _ in range(20)]
    ratings = openskill_from_results(pairs)
    s_mu, s_sigma = ratings["strong"]
    w_mu, w_sigma = ratings["weak"]
    assert ordinal(s_mu, s_sigma) > ordinal(w_mu, w_sigma)


def test_returns_mu_sigma_tuples():
    ratings = openskill_from_results([("a", "b", 1.0)])
    assert set(ratings) == {"a", "b"}
    for mu, sigma in ratings.values():
        assert isinstance(mu, float) and isinstance(sigma, float)


def test_ordinal_formula():
    assert ordinal(25.0, 8.0) == 25.0 - 3 * 8.0
