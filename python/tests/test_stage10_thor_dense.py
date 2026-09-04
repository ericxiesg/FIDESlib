"""Stage 10: the attention dense layer, and the representation change it performs.

This is the first stage whose input and output blockings differ - twelve blocks of 64 in, six of 128
out - so it is also the test that the geometry model handles that at all. The product itself is exact,
so it is checked slot by slot against `context @ W.T + 2 * b`.
"""
import numpy as np
import pytest

from thorfhe import (THOR_ATTENTION_DENSE, THOR_BERT, ClearEngine, block_diagonal_masks,
                     encode_bias_raw, encode_weight_raw)
from thorfhe.dense import DenseStages, dense_fold_mask, used_block_mask

D = THOR_ATTENTION_DENSE   # the working geometry of stage 10
Q = THOR_BERT              # the geometry stage 08 leaves its output in
DEPTH = 40


@pytest.fixture(scope="module")
def masks():
    return block_diagonal_masks(D)


def build(masks, depth=DEPTH):
    engine = ClearEngine(D, depth=depth)
    low, high = masks
    return engine, DenseStages(engine, D, masks=low, complement_masks=high)


def encode_context(engine, context):
    """The layout stage 08 leaves its output in: two complex ciphertexts of twelve blocks of 64."""
    out = []
    for ct in range(2):
        msg = np.zeros(Q.slot_count, dtype=complex)
        for group in range(Q.pack):
            for token in range(Q.dim):
                for block in range(Q.n_blocks):
                    real = (ct * Q.pack + group + token) % Q.n_out
                    imag = ((ct + 2) * Q.pack + group + token) % Q.n_out
                    msg[Q.slot(group, token, block)] = (context[token, Q.n_out * block + real]
                                                        + 1j * context[token, Q.n_out * block + imag])
        out.append(engine.encrypt(msg))
    return np.array(out, dtype=object)


def encode_layer(w, b):
    weight = encode_weight_raw(w, dim=D.dim, pack=D.pack, n_slot=D.n_slot, group_size=D.group_size,
                               slot_count=D.slot_count, n_in=D.n_in, n_out=D.n_out,
                               slot_indices=np.arange(D.n_blocks))
    bias = encode_bias_raw(b, dim=D.dim, pack=D.pack, n_slot=D.n_slot, group_size=D.group_size,
                           slot_count=D.slot_count, n_out=D.n_out, n_blocks=D.out_blocks,
                           slot_indices=np.arange(D.out_blocks))
    return weight, bias


def test_geometry_describes_a_representation_change():
    """Twelve blocks of 64 in, six of 128 out - the numbers stage 10's weight array asserts."""
    assert (D.n_output_ciphertexts, D.diag_count, D.n_in_complex) == (8, 6, 32)
    assert D.n_input_ciphertexts == 2
    assert (D.in_blocks, D.out_blocks) == (12, 6)
    assert D.n_blocks == 12  # the window the partial products are recombined in


def test_attention_dense_computes_the_linear_layer(masks):
    """`out[ct][group, t, b] = (context @ W.T + 2b)[t, n_out*b + (ct*pack + group + t) % n_out]`."""
    rng = np.random.default_rng(71)
    context = rng.normal(size=(Q.dim, Q.features)) * 0.1
    w = rng.normal(size=(Q.features, Q.features)) * 0.05
    b = rng.normal(size=(Q.features,)) * 0.1

    engine, stages = build(masks)
    weight, bias = encode_layer(w, b)
    rotated = stages.stage_02_make_rotated_copies(encode_context(engine, context))
    out = stages.stage_10_attention_dense(rotated, weight, bias)

    assert out.shape == (8,)
    want = context @ w.T + 2 * b
    got = np.stack([np.asarray(engine.decrypt(o), dtype=complex).real for o in out])

    error = 0.0
    for ct in range(8):
        for group in range(D.pack):
            diagonal = ct * D.pack + group
            for token in range(D.dim):
                for block in range(D.out_blocks):
                    error = max(error, abs(got[ct, D.slot(group, token, block)]
                                           - want[token, D.n_out * block + (diagonal + token) % D.n_out]))
    assert error < 1e-12


def test_only_the_folded_blocks_carry_the_answer(masks):
    """Blocks 6 to 11 hold the fold's leftovers; consumers must mask, and this says so.

    THOR does not clear them, so a downstream stage that reads all twelve slots gets a half-summed
    value rather than a zero - which is much harder to notice than a crash.
    """
    rng = np.random.default_rng(72)
    context = rng.normal(size=(Q.dim, Q.features)) * 0.1
    w = rng.normal(size=(Q.features, Q.features)) * 0.05

    engine, stages = build(masks)
    weight, bias = encode_layer(w, np.zeros(Q.features))
    rotated = stages.stage_02_make_rotated_copies(encode_context(engine, context))
    out = stages.stage_10_attention_dense(rotated, weight, bias)

    leftover = used_block_mask(D) == 0
    residue = max(np.abs(np.asarray(engine.decrypt(o), dtype=complex)[leftover]).max() for o in out)
    assert residue > 1e-6, "the leftovers are not zero, which is exactly why the mask is needed"
    assert dense_fold_mask(D).sum() + used_block_mask(D).sum() == D.slot_count
