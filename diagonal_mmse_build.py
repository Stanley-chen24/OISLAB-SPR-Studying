# Build the MMSE normal-equation matrices Hr, HR two ways (paper blocks vs A^T A) and compare them.
import numpy as np


def bilinear_weights(near=2/3, far=1/3):
    return dict(a1=near, b1=far, a2=near, b2=far,
                a3=near*near, b3=far*far, a4=near*near, b4=far*far,
                g1=near*far, g2=near*far, g3=near*far, g4=near*far)


def interp_mask(w):
    a1, b1, a2, b2 = w['a1'], w['b1'], w['a2'], w['b2']
    a3, b3, a4, b4 = w['a3'], w['b3'], w['a4'], w['b4']
    g1, g2, g3, g4 = w['g1'], w['g2'], w['g3'], w['g4']
    return np.array([[b3, g2, b2, g4, b4],
                     [g3, a3, a2, a4, g1],
                     [b1, a1, 1., a1, b1],
                     [g1, a4, a2, a3, g3],
                     [b4, g4, b2, g2, b3]])


def k_coeffs(w):
    a1, b1, a2, b2 = w['a1'], w['b1'], w['a2'], w['b2']
    a3, b3, a4, b4 = w['a3'], w['b3'], w['a4'], w['b4']
    g1, g2, g3, g4 = w['g1'], w['g2'], w['g3'], w['g4']
    k0 = 1 + 2*(a1**2 + b1**2 + a2**2 + b2**2 + a3**2 + b3**2 + a4**2 + b4**2
                + g1**2 + g2**2 + g3**2 + g4**2)
    k1 = 2*(a1*b1 + a3*g1 + b3*g4 + a4*g3 + b4*g2)
    k2 = 2*(a2*b2 + a3*g3 + b3*g2 + a4*g4 + b4*g1)
    k3 = 2*(a3*b3 + g1*g2)
    k4 = 2*(a4*b4 + g1*g4)
    return k0, k1, k2, k3, k4


def circ_NxN(first_row):
    fr = np.asarray(first_row, float)
    N = len(fr)
    idx = (np.arange(N)[None, :] - np.arange(N)[:, None]) % N
    return fr[idx]


def hr_block(diag, sup, sub, N):
    fr = np.zeros(N)
    fr[0], fr[1 % N], fr[(N - 1) % N] = diag, sup, sub
    return circ_NxN(fr)


def hR_block(sten5, N):
    B = np.zeros((N, 3*N))
    for p in range(N):
        for off in range(-2, 3):
            B[p, (3*p + off) % (3*N)] = sten5[off + 2]
    return B


def build_Hr_block(k, M, N):
    k0, k1, k2, k3, k4 = k
    A = hr_block(k0, k1, k1, N)
    B2 = hr_block(k2, k3, k4, N)
    B1 = hr_block(k2, k4, k3, N)
    blocks = {0: A, 1: B2, (M - 1): B1}
    Hr = np.zeros((M*N, M*N))
    for I in range(M):
        for J in range(M):
            blk = blocks.get((J - I) % M)
            if blk is not None:
                Hr[I*N:(I + 1)*N, J*N:(J + 1)*N] = blk
    return Hr, dict(A=A, B1=B1, B2=B2)


def build_HR_block(w, M, N):
    mask = interp_mask(w)
    C, D1, D2, E1, E2 = (hR_block(mask[r], N) for r in (2, 1, 3, 0, 4))
    sten = {-2: E1, -1: D1, 0: C, 1: D2, 2: E2}
    M3 = 3*M
    HR = np.zeros((M*N, 9*M*N))
    for I in range(M):
        for Jb in range(M3):
            off = (Jb - 3*I) % M3
            if off > M3 // 2:
                off -= M3
            blk = sten.get(off)
            if blk is not None:
                HR[I*N:(I + 1)*N, Jb*3*N:(Jb + 1)*3*N] = blk
    return HR, dict(C=C, D1=D1, D2=D2, E1=E1, E2=E2)


def build_A_dense(w, M, N):
    mask = interp_mask(w)
    n_rows, n_cols = 3*M, 3*N
    A = np.zeros((n_rows*n_cols, M*N))
    for i in range(M):
        for j in range(N):
            for kr in range(-2, 3):
                for kc in range(-2, 3):
                    A[((3*i + kr) % n_rows)*n_cols + (3*j + kc) % n_cols, i*N + j] += mask[kr + 2, kc + 2]
    return A


def main():
    M, N = 3, 4
    w = bilinear_weights()
    k = k_coeffs(w)
    print("k coefficients (bilinear, near=2/3, far=1/3):")
    for name, v in zip(["k0", "k1", "k2", "k3", "k4"], k):
        print(f"   {name} = {v:.6f}")
    print(f"   k0 == sum(mask^2)? {np.isclose(k[0], (interp_mask(w)**2).sum())}")

    Hr_b, hb = build_Hr_block(k, M, N)
    HR_b, _ = build_HR_block(w, M, N)
    A = build_A_dense(w, M, N)

    print(f"\nMethod A (paper blocks) vs Method B (A^T A), M={M}, N={N}:")
    print(f"   ||Hr_block - A^T A||_inf = {np.abs(Hr_b - A.T @ A).max():.1e}")
    print(f"   ||HR_block - A^T  ||_inf = {np.abs(HR_b - A.T).max():.1e}")
    print(f"   Hr symmetric?  {np.allclose(Hr_b, Hr_b.T)}")
    print(f"   B1 == B2^T ?   {np.allclose(hb['B1'], hb['B2'].T)}")
    print(f"   A  shape {A.shape},  Hr shape {Hr_b.shape},  HR shape {HR_b.shape}")

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print("\nblock A (NxN) =\n", hb['A'])
    print("\nblock B2 (NxN) =\n", hb['B2'])


if __name__ == "__main__":
    main()
