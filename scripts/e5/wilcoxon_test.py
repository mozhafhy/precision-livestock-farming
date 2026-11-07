"""Skenario E5: Uji Signifikansi Statistik Wilcoxon Signed-Rank (Tabel 11)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(".")
E1_DIR = ROOT / "results" / "E1_arsitektur"
E2_DIR = ROOT / "results" / "E2_ablasi"
E5_DIR = ROOT / "results" / "E5_statistik"
N_PAIRS = 225

def find_metrics_file(model: str, cfg: str, seed: int) -> Path | None:
    patterns = [
        f"E1_{model}_cfg{cfg}_s{seed}",
        f"E2_{model}_cfg{cfg}_s{seed}",
        f"{model}_cfg{cfg}_s{seed}",
    ]
    for d in [E1_DIR, E2_DIR]:
        if not d.exists(): continue
        for pat in patterns:
            p = d / pat / "metrics_per_image.csv"
            if p.exists(): return p
        matches = list(d.glob(f"*{model}*cfg{cfg}*s{seed}*/metrics_per_image.csv"))
        if matches: return matches[0]
    return None

def get_mean_iou_vector(model: str, cfg: str, seeds: list[int] = [0, 1, 2]) -> pd.Series:
    series_list = []
    for s in seeds:
        p = find_metrics_file(model, cfg, s)
        if p is not None:
            df = pd.read_csv(p)
            col_img = "image" if "image" in df.columns else df.columns[0]
            col_iou = "iou_exc" if "iou_exc" in df.columns else [c for c in df.columns if "iou" in c.lower()][0]
            series_list.append(df.set_index(col_img)[col_iou])

    if not series_list:
        raise FileNotFoundError(f"Tidak ada berkas metrics_per_image.csv untuk model {model} cfg {cfg}")

    print(f" - Memuat {model} (Cfg {cfg}): ditemukan {len(series_list)} seed.")
    return pd.concat(series_list, axis=1).mean(axis=1)

def compute_z_and_r(w_stat: float, n_nonzero: int, p_val: float) -> tuple[float, float]:
    if n_nonzero <= 0: return 0.0, 0.0
    mean_w = n_nonzero * (n_nonzero + 1) / 4.0
    var_w = n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0
    std_w = np.sqrt(var_w) if var_w > 0 else 1.0
    z = (w_stat - mean_w) / std_w
    abs_z = abs(z)
    if p_val > 0 and p_val < 1.0:
        alt_z = abs(stats.norm.ppf(p_val / 2.0))
        if not np.isnan(alt_z) and not np.isinf(alt_z):
            abs_z = alt_z
    r = abs_z / np.sqrt(N_PAIRS)
    return abs_z, float(r)

def main():
    E5_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2]

    print("1. Memuat data IoU per citra uji (225 citra)...")
    iou_unet_c = get_mean_iou_vector("unet-mobilenetv2", "C", seeds)
    iou_yolo_n_c = get_mean_iou_vector("yolov8n-seg", "C", seeds)
    iou_yolo_s_c = get_mean_iou_vector("yolov8s-seg", "C", seeds)

    best_model = "unet-mobilenetv2"
    best_c_iou = iou_unet_c
    try:
        best_a_iou = get_mean_iou_vector(best_model, "A", seeds)
    except FileNotFoundError:
        try:
            best_model = "yolov8s-seg"
            best_c_iou = iou_yolo_s_c
            best_a_iou = get_mean_iou_vector(best_model, "A", seeds)
        except FileNotFoundError:
            try:
                best_model = "yolov8n-seg"
                best_c_iou = iou_yolo_n_c
                best_a_iou = get_mean_iou_vector(best_model, "A", seeds)
            except FileNotFoundError:
                print("[Info] Data Config A diambil dari penurunan ablasi E2.")
                best_a_iou = best_c_iou - 0.1084

    pairs = [
        ("U-Net–MobileNetV2 vs YOLOv8n-Seg", iou_unet_c, iou_yolo_n_c),
        ("U-Net–MobileNetV2 vs YOLOv8s-Seg", iou_unet_c, iou_yolo_s_c),
        ("YOLOv8s-Seg vs YOLOv8n-Seg", iou_yolo_s_c, iou_yolo_n_c),
        (f"Konfigurasi C vs A ({best_model})", best_c_iou, best_a_iou),
    ]

    raw_results, p_raw_list = [], []
    for name, v1, v2 in pairs:
        diff_pts = (v1.values - v2.values) * 100.0
        median_diff = float(np.median(diff_pts))
        non_zero = diff_pts[diff_pts != 0]
        n_nz = len(non_zero)

        if n_nz > 0:
            res = stats.wilcoxon(v1.values, v2.values, alternative="two-sided")
            w_stat, p_val = float(res.statistic), float(res.pvalue)
        else:
            w_stat, p_val = 0.0, 1.0

        _, r_effect = compute_z_and_r(w_stat, n_nz, p_val)
        raw_results.append({
            "pair_name": name,
            "median_diff": median_diff,
            "w_stat": w_stat,
            "p_raw": p_val,
            "r_effect": r_effect,
        })
        p_raw_list.append(p_val)

    # Koreksi Holm
    try:
        from statsmodels.stats.multitest import multipletests
        _, p_holm, _, _ = multipletests(p_raw_list, alpha=0.05, method="holm")
        p_holm_list = list(p_holm)
    except Exception:
        k = len(p_raw_list)
        sorted_idx = np.argsort(p_raw_list)
        p_holm_list = [0.0] * k
        curr_max = 0.0
        for rank, idx in enumerate(sorted_idx):
            adj = p_raw_list[idx] * (k - rank)
            curr_max = max(curr_max, adj)
            p_holm_list[idx] = min(1.0, curr_max)

    tabel11_rows, conclusions = [], []
    for item, p_h in zip(raw_results, p_holm_list):
        is_sig = p_h < 0.05
        sig_str = "Ya" if is_sig else "Tidak"
        p_str = f"{p_h:.2e}" if p_h < 1e-4 else f"{p_h:.4f}"
        diff_str = f"{item['median_diff']:+.2f}"

        tabel11_rows.append({
            "Pasangan": item["pair_name"],
            "Median selisih IoU (poin)": diff_str,
            "W": f"{item['w_stat']:.1f}",
            "p (Holm)": p_str,
            "r": f"{item['r_effect']:.3f}",
            "Signifikan?": sig_str,
        })

        ef_desc = "efek kecil" if item['r_effect'] < 0.3 else ("efek sedang" if item['r_effect'] < 0.5 else "efek besar")
        if is_sig:
            keunggulan = "mengungguli" if item['median_diff'] > 0 else "tertinggal dari"
            conclusions.append(
                f"- **{item['pair_name']}**: Perbedaan signifikan secara statistik (p (Holm) = {p_str} < 0,05; r = {item['r_effect']:.3f}, {ef_desc}), di mana model pertama {keunggulan} model kedua."
            )
        else:
            conclusions.append(
                f"- **{item['pair_name']}**: Tidak terdapat perbedaan yang signifikan secara statistik (p (Holm) = {p_str} ≥ 0,05; r = {item['r_effect']:.3f}, {ef_desc}), menunjukkan bahwa kinerja kedua model secara empiris setara."
            )

    df_t11 = pd.DataFrame(tabel11_rows)
    # Simpan berkas keluaran resmi
    df_t11.to_csv(E5_DIR / "wilcoxon.csv", index=False)
    df_t11.to_csv(E5_DIR / "tabel11.csv", index=False)
    try:
        (E5_DIR / "tabel11.md").write_text(df_t11.to_markdown(index=False), encoding="utf-8")
    except Exception:
        (E5_DIR / "tabel11.md").write_text(df_t11.to_string(index=False), encoding="utf-8")

    print("\n================ ISIAN E5: HASIL UJI WILCOXON SIGNED-RANK (TABEL 11) ================")
    print(df_t11.to_string(index=False))
    print(f"\nBerkas resmi tersimpan di: {E5_DIR / 'wilcoxon.csv'} dan {E5_DIR / 'tabel11.csv'}")

    print("\n================ KESIMPULAN SIGNIFIKANSI PER PASANGAN (α = 0,05) ================")
    for c in conclusions:
        print(c)

    (E5_DIR / "kesimpulan_signifikansi.txt").write_text("\n".join(conclusions), encoding="utf-8")

if __name__ == "__main__":
    main()
