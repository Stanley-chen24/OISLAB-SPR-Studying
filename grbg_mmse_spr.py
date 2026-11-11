# Full MMSE-SD pipeline on a GRBG Bayer panel (R, B solved alone; G1, G2 solved jointly), plus DDSD.
import numpy as np
from scipy.ndimage import convolve

import diagonal_mmse_spr as diag
from grbg_mmse_build import PERIOD, mask_RB, mask_G, LAYOUT, CHANNEL, build_A_grbg, build_A_G
from diagonal_mmse_spr import solve_channel, save_result, pick_image_path, image_tag, load_input, rms


def mmse_solve(input_img):
    M, N = input_img.shape[0] // PERIOD, input_img.shape[1] // PERIOD
    A_R, A_B, A_G = build_A_grbg("R", M, N), build_A_grbg("B", M, N), build_A_G(M, N)
    d_R, res_R = solve_channel(A_R, input_img[:, :, 0].ravel())
    d_G, res_G = solve_channel(A_G, input_img[:, :, 1].ravel())
    d_B, res_B = solve_channel(A_B, input_img[:, :, 2].ravel())
    driving_signals = {"G1": d_G[:M * N].reshape(M, N), "R": d_R.reshape(M, N),
                       "B": d_B.reshape(M, N), "G2": d_G[M * N:].reshape(M, N)}
    V_imp = np.stack([A_R @ d_R, A_G @ d_G, A_B @ d_B], axis=-1).reshape(PERIOD * M, PERIOD * N, 3)
    return driving_signals, V_imp, max(res_R, res_G, res_B)


def place_panel(driving_signals):
    return diag.place_panel(driving_signals, LAYOUT, CHANNEL)


def reconstruct(D):
    return np.stack([convolve(D[:, :, 0], mask_RB, mode='wrap'),
                     convolve(D[:, :, 1], mask_G, mode='wrap'),
                     convolve(D[:, :, 2], mask_RB, mode='wrap')], axis=-1)


def ddsd_downsampling(input_img):
    return diag.ddsd_downsampling(input_img, LAYOUT, CHANNEL)


def main():
    path = pick_image_path()
    tag = image_tag(path)
    input_img = load_input(path)
    driving_signals, V_imp, res = mmse_solve(input_img)
    D = place_panel(driving_signals)
    V = reconstruct(D)
    D_ddsd = place_panel(ddsd_downsampling(input_img))
    V_ddsd = reconstruct(D_ddsd)

    print(f"driving_signals : d_R, d_B {driving_signals['R'].shape}, d_G1, d_G2 {driving_signals['G1'].shape}  (not an image, not saved)")
    print(f"panel   D       : {D.shape}")
    print(f"virtual V       : {V.shape}")
    print(f"normal-eq residual max|A^T(input - A d)| (before clip) = {res:.1e}")
    print(f"explicit mask*D vs implicit A d : max diff = {np.abs(V - V_imp).max():.1e}")
    print(f"MMSE : max|V-input| = {np.abs(V - input_img).max():.3f}   rms = {rms(V - input_img):.4f}")
    print(f"DDSD : max|V-input| = {np.abs(V_ddsd - input_img).max():.3f}   rms = {rms(V_ddsd - input_img):.4f}")
    print(f"MMSE rms <= DDSD rms ? {rms(V - input_img) <= rms(V_ddsd - input_img) + 1e-12}")
    print()
    save_result(D, tag, "GRBG", "MMSE", "panel")
    save_result(V, tag, "GRBG", "MMSE", "virtual")
    save_result(D_ddsd, tag, "GRBG", "DDSD", "panel")
    save_result(V_ddsd, tag, "GRBG", "DDSD", "virtual")


if __name__ == "__main__":
    main()
