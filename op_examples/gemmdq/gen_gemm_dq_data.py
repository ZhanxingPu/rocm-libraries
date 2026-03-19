#!/usr/bin/env python3
"""
GemmDq test data generator + NumPy reference

Generates random FP16 A, UINT4-packed B, scales, zeros, and computes
the golden reference C = A @ dequant(B)^T using NumPy.

Usage:
    python3 gen_gemm_dq_data.py [MxKxN] [--group-size GS] [--dir DIR]

Examples:
    python3 gen_gemm_dq_data.py                          # 128x128x128 gs=128
    python3 gen_gemm_dq_data.py 256x512x256 --group-size 128
"""

import numpy as np
import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser(description='Generate GemmDq test data')
    parser.add_argument('size', nargs='?', type=str, default='128x128x128',
                        help='Matrix size MxKxN (default: 128x128x128)')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Quantization group size along K (default: 128)')
    parser.add_argument('--no-ref', action='store_true',
                        help='Skip computing reference C')
    parser.add_argument('--dir', type=str, default='data',
                        help='Output directory (default: data/)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    parts = args.size.split('x')
    if len(parts) != 3:
        parser.error(f"Size must be MxKxN (got '{args.size}')")
    M, K, N = int(parts[0]), int(parts[1]), int(parts[2])
    group_size = args.group_size
    num_groups_k = K // group_size

    if K % group_size != 0:
        parser.error(f"K ({K}) must be divisible by group_size ({group_size})")
    if K % 2 != 0:
        parser.error(f"K ({K}) must be even for uint4 packing")

    out_dir = args.dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"GemmDq Data Generator")
    print(f"  M={M}, N={N}, K={K}, group_size={group_size}")
    print(f"  num_groups_k={num_groups_k}")
    print(f"  A: FP16 col-major (M x K)")
    print(f"  B_packed: UINT4 (N x K/2 bytes)")
    print(f"  scales/zeros: FP16 (N x num_groups_k)")
    print(f"  C: FP16 col-major (M x N)")
    print(f"  Seed: {args.seed}")
    print(f"  Output dir: {os.path.abspath(out_dir)}")
    print()

    np.random.seed(args.seed)

    # ---- A: FP16 col-major (M x K) ----
    print("Generating A (FP16)...", end=" ", flush=True)
    A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float16)
    print(f"shape={A.shape}, {A.nbytes / 1024:.1f} KB")

    # ---- B: random uint4 values (N x K), pack into bytes (N x K/2) ----
    print("Generating B uint4 values...", end=" ", flush=True)
    B_uint4 = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    B_even = B_uint4[:, 0::2]  # low nibble
    B_odd  = B_uint4[:, 1::2]  # high nibble
    B_packed = (B_even | (B_odd << 4)).astype(np.uint8)
    print(f"packed shape={B_packed.shape}, {B_packed.nbytes / 1024:.1f} KB")

    # ---- scales: FP16 (N x num_groups_k) ----
    print("Generating scales (FP16)...", end=" ", flush=True)
    scales = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    print(f"shape={scales.shape}")

    # ---- zeros: FP16 (N x num_groups_k) ----
    print("Generating zeros (FP16)...", end=" ", flush=True)
    zeros = np.random.uniform(7.0, 9.0, (N, num_groups_k)).astype(np.float16)
    print(f"shape={zeros.shape}")

    # ---- Save binary files ----
    # A: col-major (Fortran order)
    file_A = os.path.join(out_dir, "gemm_dq_A.bin")
    A.flatten(order='F').tofile(file_A)
    print(f"\nSaved {file_A} ({os.path.getsize(file_A) / 1024:.1f} KB)")

    # B_packed: row-major (C order) => layout [n * (K/2) + k/2]
    file_B = os.path.join(out_dir, "gemm_dq_B_packed.bin")
    B_packed.flatten(order='C').tofile(file_B)
    print(f"Saved {file_B} ({os.path.getsize(file_B) / 1024:.1f} KB)")

    # scales: row-major => layout [n * num_groups_k + g]
    file_S = os.path.join(out_dir, "gemm_dq_scales.bin")
    scales.flatten(order='C').tofile(file_S)
    print(f"Saved {file_S} ({os.path.getsize(file_S) / 1024:.1f} KB)")

    # zeros: row-major => layout [n * num_groups_k + g]
    file_Z = os.path.join(out_dir, "gemm_dq_zeros.bin")
    zeros.flatten(order='C').tofile(file_Z)
    print(f"Saved {file_Z} ({os.path.getsize(file_Z) / 1024:.1f} KB)")

    # ---- Compute reference: C = A @ dequant(B)^T ----
    if not args.no_ref:
        print(f"\nComputing reference C = A @ dequant(B)^T (FP32 accumulation)...",
              flush=True)
        t0 = time.time()

        # Dequantize: B_dq[n, k] = (B_uint4[n, k] - zeros[n, k//gs]) * scales[n, k//gs]
        group_idx = np.arange(K) // group_size  # shape (K,)
        scales_f32 = scales.astype(np.float32)  # (N, num_groups_k)
        zeros_f32  = zeros.astype(np.float32)   # (N, num_groups_k)

        B_dq = (B_uint4.astype(np.float32) - zeros_f32[:, group_idx]) \
             * scales_f32[:, group_idx]  # (N, K)

        # C = A @ B_dq^T => (M, K) @ (K, N) = (M, N)
        C_ref_f32 = A.astype(np.float32) @ B_dq.T
        C_ref = C_ref_f32.astype(np.float16)

        elapsed = time.time() - t0
        gflops = (2.0 * M * N * K) / (elapsed * 1e9)
        print(f"  Time: {elapsed:.3f} s ({gflops:.1f} GFLOPS on CPU)")

        # Save C_ref col-major (Fortran order) as FP16
        file_C = os.path.join(out_dir, "gemm_dq_C_ref.bin")
        C_ref.flatten(order='F').tofile(file_C)
        print(f"Saved {file_C} ({os.path.getsize(file_C) / 1024:.1f} KB)")

        print(f"\n  C[0,0] = {float(C_ref[0, 0]):.4f}")
        if M > 1 and N > 1:
            print(f"  C[0,1] = {float(C_ref[0, 1]):.4f}")
            print(f"  C[1,0] = {float(C_ref[1, 0]):.4f}")
            print(f"  C[{M-1},{N-1}] = {float(C_ref[M-1, N-1]):.4f}")
        print(f"  C mean = {float(C_ref.mean()):.4f}")
        print(f"  C max  = {float(C_ref.max()):.4f}")
        print(f"  C min  = {float(C_ref.min()):.4f}")
    else:
        print("\nSkipping reference computation (--no-ref)")

    # ---- Metadata ----
    meta_file = os.path.join(out_dir, "gemm_dq_meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"M={M}\n")
        f.write(f"N={N}\n")
        f.write(f"K={K}\n")
        f.write(f"group_size={group_size}\n")
        f.write(f"num_groups_k={num_groups_k}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"A_layout=col-major\n")
        f.write(f"C_layout=col-major\n")
        f.write(f"B_packed_layout=row-major\n")
        f.write(f"scales_zeros_layout=row-major\n")

    print(f"\nFiles generated in {os.path.abspath(out_dir)}/:")
    for fn in ["gemm_dq_A.bin", "gemm_dq_B_packed.bin", "gemm_dq_scales.bin",
               "gemm_dq_zeros.bin", "gemm_dq_C_ref.bin", "gemm_dq_meta.txt"]:
        fp = os.path.join(out_dir, fn)
        if os.path.exists(fp):
            sz = os.path.getsize(fp)
            if sz > 1024 * 1024:
                print(f"  {fn:30s}  {sz / 1024 / 1024:8.1f} MB")
            else:
                print(f"  {fn:30s}  {sz / 1024:8.1f} KB")
        else:
            print(f"  {fn:30s}  (not generated)")

    print(f"\nDone! Now run:")
    print(f"  make && make run ARGS=\"{M}x{K}x{N} {group_size}\"")


if __name__ == '__main__':
    main()
