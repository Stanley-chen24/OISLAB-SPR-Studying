# GRBG normal equations, Method A (mask autocorrelation stencils) vs Method B (A^T A), incl. the coupled G1/G2 block.
import numpy as np
import scipy.sparse as sp

import diagonal_mmse_spr as diag

PERIOD = 2
mask_RB = np.array([[.25, .5, .25],
                    [.5, 1., .5],
                    [.25, .5, .25]])
mask_G = np.array([[0., .25, 0.],
                   [.25, 1., .25],
                   [0., .25, 0.]])

LAYOUT = {"G1": np.array([[1, 0], [0, 0]]), "R": np.array([[0, 1], [0, 0]]),
          "B": np.array([[0, 0], [1, 0]]), "G2": np.array([[0, 0], [0, 1]])}
CHANNEL = {"G1": 1, "R": 0, "B": 2, "G2": 1}
MASK = {"G1": mask_G, "R": mask_RB, "B": mask_RB, "G2": mask_G}

STENCIL_RB = {(0, 0): 2.25,
              (0, 1): .375, (0, -1): .375, (1, 0): .375, (-1, 0): .375,
              (1, 1): .0625, (1, -1): .0625, (-1, 1): .0625, (-1, -1): .0625}
STENCIL_G_SELF = {(0, 0): 1.25,
                  (0, 1): .0625, (0, -1): .0625, (1, 0): .0625, (-1, 0): .0625}
STENCIL_G_CROSS = {(0, 0): .125, (0, -1): .125, (-1, 0): .125, (-1, -1): .125}


def build_A_grbg(k, M, N):
    return diag.build_A_sparse(diag.offset_of(LAYOUT[k]), M, N, MASK[k], period=PERIOD)


def build_A_G(M, N):
    return sp.hstack([build_A_grbg("G1", M, N), build_A_grbg("G2", M, N)]).tocsr()


def block_circulant(stencil, M, N):
    row_idx, col_idx, vals = [], [], []
    for i in range(M):
        for j in range(N):
            for (si, sj), w in stencil.items():
                row_idx.append(i * N + j)
                col_idx.append(((i + si) % M) * N + (j + sj) % N)
                vals.append(w)
    return sp.csr_matrix((vals, (row_idx, col_idx)), shape=(M * N, M * N))


def method_a_G(M, N):
    self_blk = block_circulant(STENCIL_G_SELF, M, N)
    cross = block_circulant(STENCIL_G_CROSS, M, N)
    return sp.bmat([[self_blk, cross], [cross.T, self_blk]]).tocsr()


def maxdiff(P, Q):
    return float(np.abs((P - Q).toarray()).max())


def main():
    M, N = 3, 4
    print(f"GRBG cross-validation (M={M}, N={N}, input {PERIOD * M}x{PERIOD * N})")
    print(f"mask_RB sum = {mask_RB.sum():.3f}, x 1/4 = {mask_RB.sum() / 4:.3f}  (DC gain)")
    print(f"mask_G  sum = {mask_G.sum():.3f}, x 1/2 = {mask_G.sum() / 2:.3f}  (DC gain)")
    print()
    for k in ["R", "B"]:
        A = build_A_grbg(k, M, N)
        print(f"[{k}] ||Hr_block - A^T A||_inf = {maxdiff(block_circulant(STENCIL_RB, M, N), A.T @ A):.1e}")
    print(f"[R] k0 == sum(mask_RB^2)? {np.isclose(STENCIL_RB[(0, 0)], (mask_RB ** 2).sum())}")
    print()
    A_G = build_A_G(M, N)
    A_G1, A_G2 = A_G[:, :M * N], A_G[:, M * N:]
    print(f"[G] A_G shape {A_G.shape}  (2MN columns -> joint solve)")
    print(f"[G] self-block k0 == sum(mask_G^2)? {np.isclose(STENCIL_G_SELF[(0, 0)], (mask_G ** 2).sum())}")
    print(f"[G] ||Hr_block - A_G^T A_G||_inf = {maxdiff(method_a_G(M, N), A_G.T @ A_G):.1e}")
    print(f"[G] self  block A_G1^T A_G1 vs Method A : {maxdiff(A_G1.T @ A_G1, block_circulant(STENCIL_G_SELF, M, N)):.1e}")
    print(f"[G] cross block A_G1^T A_G2 vs Method A : {maxdiff(A_G1.T @ A_G2, block_circulant(STENCIL_G_CROSS, M, N)):.1e}")
    print(f"[G] cross block max entry = {abs(A_G1.T @ A_G2).max():.3f}  (nonzero -> G1 and G2 are coupled)")


if __name__ == "__main__":
    main()
