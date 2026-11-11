# Main program: run GRBG MMSE-SD and DDSD on the same image and save both results.
import numpy as np

import grbg_mmse_spr as g


def main():
    path = g.pick_image_path()
    tag = g.image_tag(path)
    input_img = g.load_input(path)

    driving_signals, V_imp, res = g.mmse_solve(input_img)
    D_mmse = g.place_panel(driving_signals)
    V_mmse = g.reconstruct(D_mmse)
    D_ddsd = g.place_panel(g.ddsd_downsampling(input_img))
    V_ddsd = g.reconstruct(D_ddsd)

    print(f"[MMSE] normal-eq residual max|A^T(input - A d)| (before clip) = {res:.1e}")
    print(f"[MMSE] explicit mask*D vs implicit A d : max diff = {np.abs(V_mmse - V_imp).max():.1e}")
    print()
    results = [("MMSE", D_mmse, V_mmse), ("DDSD", D_ddsd, V_ddsd)]
    print("method  max|V-input|  rms")
    for name, _, V in results:
        print(f"{name:6s}  {np.abs(V - input_img).max():.3f}         {g.rms(V - input_img):.4f}")
    print()
    for name, D, V in results:
        g.save_result(D, tag, "GRBG", name, "panel")
        g.save_result(V, tag, "GRBG", name, "virtual")


if __name__ == "__main__":
    main()
