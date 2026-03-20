#!/usr/bin/env python3
"""
RoPE + GQA Flash Attention test data generator + NumPy golden reference

Pipeline: RoPE(Q, K) -> GQA(softmax(QK^T/sqrt(d)) @ V) -> O

Usage:
    python3 gen_rope_gqa_data.py                                             # defaults
    python3 gen_rope_gqa_data.py --batch 2 --seqlen_q 256 --seqlen_k 256    # custom
    python3 gen_rope_gqa_data.py --nhead_q 32 --nhead_k 8 --hdim 128        # GQA 32:8
    python3 gen_rope_gqa_data.py --rotary_dim 64                             # partial RoPE

Generates:
    data/gqa_Q.bin       - Query  [batch, nhead_q, seqlen_q, hdim]  FP16 (before RoPE)
    data/gqa_K.bin       - Key    [batch, nhead_k, seqlen_k, hdim]  FP16 (before RoPE)
    data/gqa_V.bin       - Value  [batch, nhead_k, seqlen_k, hdim]  FP16
    data/gqa_cos.bin     - cos table [max_seqlen, rotary_dim/2]     FP16
    data/gqa_sin.bin     - sin table [max_seqlen, rotary_dim/2]     FP16
    data/gqa_O_ref.bin   - Output [batch, nhead_q, seqlen_q, hdim]  FP16 (golden)
    data/gqa_meta.txt    - Metadata
"""

import numpy as np
import argparse
import os
import time


def generate_cos_sin(max_seqlen, rotary_dim):
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10000.0 ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim))
    positions = np.arange(max_seqlen, dtype=np.float64)
    angles = np.outer(positions, freqs)
    cos_table = np.cos(angles).astype(np.float16)
    sin_table = np.sin(angles).astype(np.float16)
    return cos_table, sin_table


def apply_rope(x, cos_table, sin_table, rotary_dim):
    half_rot = rotary_dim // 2
    x_f32 = x.astype(np.float32)
    cos_f32 = cos_table.astype(np.float32)
    sin_f32 = sin_table.astype(np.float32)

    seqlen = x.shape[2]
    cos_b = cos_f32[:seqlen][np.newaxis, np.newaxis, :, :]
    sin_b = sin_f32[:seqlen][np.newaxis, np.newaxis, :, :]

    x1 = x_f32[..., :half_rot]
    x2 = x_f32[..., half_rot:rotary_dim]

    result = x_f32.copy()
    result[..., :half_rot] = x1 * cos_b - x2 * sin_b
    result[..., half_rot:rotary_dim] = x1 * sin_b + x2 * cos_b
    return result.astype(np.float16)


def gqa_reference(Q, K, V, nhead_q, nhead_k, scale):
    batch, _, seqlen_q, hdim = Q.shape
    ratio = nhead_q // nhead_k
    Q_f32 = Q.astype(np.float32)
    K_f32 = K.astype(np.float32)
    V_f32 = V.astype(np.float32)
    O = np.zeros((batch, nhead_q, seqlen_q, hdim), dtype=np.float32)

    for b in range(batch):
        for hq in range(nhead_q):
            hk = hq // ratio
            S = Q_f32[b, hq] @ K_f32[b, hk].T * scale
            S_max = S.max(axis=-1, keepdims=True)
            P = np.exp(S - S_max)
            P = P / P.sum(axis=-1, keepdims=True)
            O[b, hq] = P @ V_f32[b, hk]
    return O


def main():
    parser = argparse.ArgumentParser(description='Generate RoPE + GQA test data')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--seqlen_q', type=int, default=256)
    parser.add_argument('--seqlen_k', type=int, default=256)
    parser.add_argument('--nhead_q', type=int, default=32)
    parser.add_argument('--nhead_k', type=int, default=8)
    parser.add_argument('--hdim', type=int, default=128)
    parser.add_argument('--rotary_dim', type=int, default=0,
                        help='Rotary dimension (default: hdim)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dir', type=str, default='data')
    args = parser.parse_args()

    batch = args.batch
    seqlen_q = args.seqlen_q
    seqlen_k = args.seqlen_k
    nhead_q = args.nhead_q
    nhead_k = args.nhead_k
    hdim = args.hdim
    rotary_dim = args.rotary_dim if args.rotary_dim > 0 else hdim
    out_dir = args.dir

    assert nhead_q % nhead_k == 0
    assert rotary_dim % 2 == 0
    assert rotary_dim <= hdim

    ratio = nhead_q // nhead_k
    scale = 1.0 / np.sqrt(hdim).item()
    max_seqlen = max(seqlen_q, seqlen_k)

    os.makedirs(out_dir, exist_ok=True)

    label = 'MHA' if ratio == 1 else ('MQA' if nhead_k == 1 else 'GQA')
    print("RoPE + GQA Data Generator")
    print("=" * 60)
    print(f"  batch={batch} seqlen_q={seqlen_q} seqlen_k={seqlen_k}")
    print(f"  nhead_q={nhead_q} nhead_k={nhead_k} ({label} {nhead_q}:{nhead_k})")
    print(f"  hdim={hdim} rotary_dim={rotary_dim}")
    print(f"  scale={scale:.6f}  seed={args.seed}")
    print()

    np.random.seed(args.seed)

    Q = (np.random.randn(batch, nhead_q, seqlen_q, hdim) * 0.5).astype(np.float16)
    K = (np.random.randn(batch, nhead_k, seqlen_k, hdim) * 0.5).astype(np.float16)
    V = (np.random.randn(batch, nhead_k, seqlen_k, hdim) * 0.5).astype(np.float16)

    cos_table, sin_table = generate_cos_sin(max_seqlen, rotary_dim)

    for name, arr, fn in [("Q", Q, "gqa_Q.bin"), ("K", K, "gqa_K.bin"),
                           ("V", V, "gqa_V.bin"),
                           ("cos", cos_table, "gqa_cos.bin"),
                           ("sin", sin_table, "gqa_sin.bin")]:
        path = os.path.join(out_dir, fn)
        arr.tofile(path)
        sz = os.path.getsize(path)
        print(f"  Saved {fn:20s}  {sz/1024:8.1f} KB  shape={arr.shape}")

    print(f"\nApplying RoPE + computing GQA reference...")
    t0 = time.time()
    Q_rope = apply_rope(Q, cos_table, sin_table, rotary_dim)
    K_rope = apply_rope(K, cos_table, sin_table, rotary_dim)
    O_ref = gqa_reference(Q_rope, K_rope, V, nhead_q, nhead_k, scale).astype(np.float16)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.3f} s")

    path = os.path.join(out_dir, "gqa_O_ref.bin")
    O_ref.tofile(path)
    print(f"  Saved {'gqa_O_ref.bin':20s}  {os.path.getsize(path)/1024:8.1f} KB  shape={O_ref.shape}")

    print(f"\n  O[0,0,0,:4]  = {O_ref[0,0,0,:4]}")
    print(f"  O mean={float(O_ref.mean()):.6f}  std={float(O_ref.std()):.6f}")

    meta_file = os.path.join(out_dir, "gqa_meta.txt")
    with open(meta_file, 'w') as f:
        for k, v in [("batch", batch), ("seqlen_q", seqlen_q), ("seqlen_k", seqlen_k),
                      ("nhead_q", nhead_q), ("nhead_k", nhead_k), ("hdim", hdim),
                      ("rotary_dim", rotary_dim), ("seed", args.seed),
                      ("scale", f"{scale:.8f}")]:
            f.write(f"{k}={v}\n")

    print(f"\nDone! Now run:")
    print(f"  make && make run ARGS=\"{batch} {seqlen_q} {seqlen_k} {nhead_q} {nhead_k} {hdim} data\"")


if __name__ == '__main__':
    main()
