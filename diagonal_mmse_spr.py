# Diagonal DDSD / MMSE-SD pipeline (Fang et al., TCSVT 2012): input_img -> driving_signals -> D = sum d_c kron K_c -> V = mask * D.
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import convolve
from PIL import Image

from diagonal_mmse_build import bilinear_weights, interp_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT, "result")
INPUT_SIZE = 240
PERIOD = 3
LAYOUT = {"R": np.diag([1, 0, 0]), "G": np.diag([0, 1, 0]), "B": np.diag([0, 0, 1])}
CHANNEL = {"R": 0, "G": 1, "B": 2}


def offset_of(K):
    return tuple(int(v) for v in np.argwhere(K)[0])


def build_A_sparse(offset, M, N, mask, period=PERIOD):
    rad = mask.shape[0] // 2
    n_rows, n_cols = period * M, period * N
    row_idx, col_idx, vals = [], [], []
    for i in range(M):
        for j in range(N):
            r0, c0 = period * i + offset[0], period * j + offset[1]
            for kr in range(-rad, rad + 1):
                for kc in range(-rad, rad + 1):
                    w = mask[kr + rad, kc + rad]
                    if w != 0.0:
                        row_idx.append(((r0 + kr) % n_rows) * n_cols + (c0 + kc) % n_cols)
                        col_idx.append(i * N + j)
                        vals.append(w)
    return sp.csr_matrix((vals, (row_idx, col_idx)), shape=(n_rows * n_cols, M * N))


def solve_channel(A, y):
    d = spla.spsolve((A.T @ A).tocsc(), A.T @ y)
    return np.clip(d, 0, 1), float(np.abs(A.T @ (y - A @ d)).max())


def mmse_solve(input_img, w):
    mask = interp_mask(w)
    M, N = input_img.shape[0] // PERIOD, input_img.shape[1] // PERIOD
    driving_signals, V_imp, res = {}, np.zeros_like(input_img), 0.0
    for k, K in LAYOUT.items():
        A = build_A_sparse(offset_of(K), M, N, mask)
        d, res_k = solve_channel(A, input_img[:, :, CHANNEL[k]].ravel())
        driving_signals[k] = d.reshape(M, N)
        V_imp[:, :, CHANNEL[k]] = (A @ d).reshape(PERIOD * M, PERIOD * N)
        res = max(res, res_k)
    return driving_signals, V_imp, res


def ddsd_downsampling(input_img, layout=LAYOUT, channel=CHANNEL):
    driving_signals = {}
    for k, K in layout.items():
        r0, c0 = offset_of(K)
        driving_signals[k] = input_img[r0::K.shape[0], c0::K.shape[1], channel[k]].copy()
    return driving_signals


def place_panel(driving_signals, layout=LAYOUT, channel=CHANNEL):
    k0 = next(iter(driving_signals))
    D = np.zeros(np.kron(driving_signals[k0], layout[k0]).shape + (3,))
    for k, d_k in driving_signals.items():
        D[:, :, channel[k]] += np.kron(d_k, layout[k])
    return D


def reconstruct(D, mask):
    return np.stack([convolve(D[:, :, c], mask, mode='wrap') for c in range(3)], axis=-1)


def rms(X):
    return float(np.sqrt(np.mean(X ** 2)))


def yuv_rms(X):
    Y = X @ np.array([0.299, 0.587, 0.114])
    return rms(Y), float(np.sqrt((rms(X[..., 2] - Y) ** 2 + rms(X[..., 0] - Y) ** 2) / 2))


def image_tag(path):
    return "white" if path is None else os.path.splitext(os.path.basename(path))[0]


def save_result(arr01, tag, panel, method, kind):
    path = os.path.join(RESULT_DIR, tag, panel, method, f"{tag}_{panel}_{method}_{kind}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.clip(np.round(arr01 * 255), 0, 255).astype(np.uint8)).save(path)
    print(f"saved -> {os.path.relpath(path, ROOT).replace(os.sep, '/')}")


def pick_image_path():
    if len(sys.argv) > 1:
        return None if sys.argv[1] == "--white" else sys.argv[1]
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Pick an input image (Cancel = white test image)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception as e:
        print(f"[no GUI, using white test image] {e}")
        return None


def load_input(path):
    if path is None:
        print(f"[input] white test image {INPUT_SIZE}x{INPUT_SIZE}")
        return np.ones((INPUT_SIZE, INPUT_SIZE, 3))
    img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    print(f"[input] {os.path.basename(path)} -> {INPUT_SIZE}x{INPUT_SIZE}")
    return np.asarray(img, float) / 255.0


def main():
    w = bilinear_weights()
    mask = interp_mask(w)
    path = pick_image_path()
    tag = image_tag(path)
    input_img = load_input(path)

    driving_signals, V_imp, res = mmse_solve(input_img, w)
    D = place_panel(driving_signals)
    V = reconstruct(D, mask)

    print(f"driving_signals : d_R, d_G, d_B each {driving_signals['R'].shape}  (not an image, not saved)")
    print(f"panel   D       : {D.shape}")
    print(f"virtual V       : {V.shape}")
    print(f"normal-eq residual max|A^T(input - A d)| (before clip) = {res:.1e}")
    print(f"explicit mask*D vs implicit A d : max diff = {np.abs(V - V_imp).max():.1e}")
    print()
    methods = {"DDSD": ddsd_downsampling(input_img), "MMSE": driving_signals}
    panels = {name: place_panel(ds) for name, ds in methods.items()}
    virtuals = {name: reconstruct(Dx, mask) for name, Dx in panels.items()}
    print("method  rms     luma    chroma")
    for name, Vx in virtuals.items():
        y, uv = yuv_rms(Vx - input_img)
        print(f"{name:6s}  {rms(Vx - input_img):.4f}  {y:.4f}  {uv:.4f}")
    print()
    for name in methods:
        save_result(panels[name], tag, "Diagonal", name, "panel")
        save_result(virtuals[name], tag, "Diagonal", name, "virtual")


if __name__ == "__main__":
    main()
