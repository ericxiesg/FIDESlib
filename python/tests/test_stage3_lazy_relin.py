"""Stage 3: lazy relinearisation — degree-2 ciphertexts, accumulation, masking, single relinearise."""
import numpy as np

from conftest import rand


def test_mult_relin_matches(engine):
    x, y = rand(engine, 20), rand(engine, 21)
    cx, cy = engine.encrypt(x), engine.encrypt(y)
    lazy = engine.relinearize(engine.multiply(cx, cy))
    eager = engine.multiply(cx, cy, relin=True)
    a, b = engine.decrypt_real(engine.rescale(lazy)), engine.decrypt_real(engine.rescale(eager))
    assert np.max(np.abs(a - x * y)) < 1e-5
    assert np.max(np.abs(b - x * y)) < 1e-5


def test_square_lazy(engine):
    x = rand(engine, 22)
    cx = engine.encrypt(x)
    s = engine.rescale(engine.relinearize(engine.square(cx)))
    assert np.max(np.abs(engine.decrypt_real(s) - x * x)) < 1e-5


def test_accumulate_then_relin(engine):
    """THOR CC-MM idiom: sum_k (a_k * b_k) with one relinearisation."""
    n = 4
    xs = [rand(engine, 30 + k) for k in range(n)]
    ys = [rand(engine, 40 + k) for k in range(n)]
    acc = None
    for xk, yk in zip(xs, ys):
        prod = engine.multiply(engine.encrypt(xk), engine.encrypt(yk))
        acc = prod if acc is None else engine.add(acc, prod)
    out = engine.rescale(engine.relinearize(acc))
    expect = sum(xk * yk for xk, yk in zip(xs, ys))
    assert np.max(np.abs(engine.decrypt_real(out) - expect)) < 1e-5


def test_degree2_masking_and_sub(engine):
    """THOR stage_06: multiplied -> rescale -> mask (pt mult) -> subtract -> later relin."""
    x, y, m = rand(engine, 50), rand(engine, 51), (np.arange(engine.slots) % 2).astype(float)
    prod = engine.rescale(engine.multiply(engine.encrypt(x), engine.encrypt(y)))
    masked = engine.rescale(engine.multiply(prod, m))
    rest = engine.subtract(engine.level_down(prod, 1), masked)  # explicit level alignment, as he.py does
    got = engine.decrypt_real(engine.relinearize(rest))
    assert np.max(np.abs(got - x * y * (1 - m))) < 1e-4
