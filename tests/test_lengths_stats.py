import random, pytest
from cot_compress.lengths import code_points, NGramCoder, adaptive_bits, zstd_ratio
from cot_compress.stats import permutation_test, seed_ci

def test_code_points_not_bytes():
    assert code_points("ab") == 2 and code_points("中文") == 2 and len("中文".encode()) == 6

def test_ngram_coder_learns_structure():
    rng = random.Random(0)
    periodic = [[1, 2, 3, 4, 5] * 20 for _ in range(20)]
    noise = [[rng.randrange(1000) for _ in range(100)] for _ in range(20)]
    bp = adaptive_bits(periodic, vocab_size=1000, seed=0); bn = adaptive_bits(noise, vocab_size=1000, seed=0)
    assert bp["bits_per_token"] < 1.0 < bn["bits_per_token"]
    c = NGramCoder(n=3, vocab_size=10).fit([[1, 2, 3, 1, 2, 3]])
    assert c.prob([1, 2], 3) > c.prob([1, 2], 9) and 0 < c.prob([], 7) < 1
    assert adaptive_bits([[1, 2, 3]], 10)["bits_mean"] is None      # 分不了半

def test_zstd_ratio():
    r, name = zstd_ratio("7 " * 500); assert r is not None and r < 0.2 and name in ("zstd19", "zlib9")
    assert zstd_ratio("short")[0] is None

def test_permutation_test():
    r = permutation_test([1.0, 1.1, 0.9, 1.2, 1.0], [0.2, 0.3, 0.1, 0.25, 0.2])
    assert r["exact"] and r["p"] < 0.05 and r["diff"] > 0
    r2 = permutation_test([1.0, 1.1, 0.9], [1.0, 1.05, 0.95]); assert r2["p"] > 0.2
    r3 = permutation_test(list(range(20)), list(range(20)), n_perm=500); assert not r3["exact"] and r3["n_perm"] == 500
    ci = seed_ci([0.5, 0.6, 0.7]); assert ci["lo"] < 0.6 < ci["hi"]
    assert seed_ci([0.5])["se"] is None
