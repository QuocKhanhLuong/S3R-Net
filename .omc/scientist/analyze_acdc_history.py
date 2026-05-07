#!/usr/bin/env python3
"""Analyze acdc_asym_v3_c48_history.json — stdlib only."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "weights" / "acdc_asym_v3_c48_history.json"
REPORT_DIR = ROOT / ".omc" / "scientist" / "reports"
FIG_DIR = ROOT / ".omc" / "scientist" / "figures"


def pearson_r(xs, ys):
    n = len(xs)
    if n != len(ys) or n < 2:
        return float("nan")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def linear_slope(xs, ys):
    """Least-squares slope y ~ a + b*x."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    return a, b


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        d = json.load(f)

    ep = d["epoch"]
    train_loss = d["train_loss"]
    val_dice = d["val_dice"]
    val_hd95 = d["val_hd95"]
    lr = d["lr"]
    dice_rv = d["dice_rv"]
    dice_myo = d["dice_myo"]
    dice_lv = d["dice_lv"]
    hd95_rv = d["hd95_rv"]
    hd95_myo = d["hd95_myo"]
    hd95_lv = d["hd95_lv"]

    n = len(ep)
    assert len(set(map(len, [train_loss, val_dice, val_hd95, lr, dice_rv, dice_myo, dice_lv, hd95_rv, hd95_myo, hd95_lv]))) == 1

    lines = []
    lines.append("# ACDC asym v3 c48 — training dynamics\n")
    lines.append(f"**Source:** `{JSON_PATH}`  \n**Epochs:** {n} (epoch {ep[0]}–{ep[-1]})\n")

    # 1) Sliding window std (10 epochs)
    lines.append("\n## 1. Oscillation: std dev in 10-epoch windows\n")
    lines.append("| Window (epochs) | std(val_dice) | std(val_hd95) |\n|---|---|---|")
    w = 10
    for start in range(0, n, w):
        end = min(start + w, n)
        wd = val_dice[start:end]
        wh = val_hd95[start:end]
        e0, e1 = ep[start], ep[end - 1]
        sd = statistics.pstdev(wd) if len(wd) > 1 else 0.0
        sh = statistics.pstdev(wh) if len(wh) > 1 else 0.0
        lines.append(f"| {e0}–{e1} | {sd:.6f} | {sh:.6f} |")
    lines.append("\n*Population std (pstdev) within each window.*\n")

    # 2) Top 10 val_dice drops (epoch N minus N-1, most negative)
    dice_delta = [(ep[i], val_dice[i] - val_dice[i - 1]) for i in range(1, n)]
    dice_drops = sorted(dice_delta, key=lambda t: t[1])[:10]
    lines.append("\n## 2. Top 10 largest epoch-to-epoch **drops** in val_dice\n")
    lines.append("(Δ = dice[epoch] − dice[epoch−1]; more negative = larger drop)\n")
    lines.append("| Rank | Epoch | Δ val_dice |\n|---:|---:|---:|")
    for r, (e, delta) in enumerate(dice_drops, 1):
        lines.append(f"| {r} | {e} | {delta:.6f} |")

    # 3) Top 10 hd95 spikes (positive = worse)
    hd_delta = [(ep[i], val_hd95[i] - val_hd95[i - 1]) for i in range(1, n)]
    hd_spikes = sorted(hd_delta, key=lambda t: t[1], reverse=True)[:10]
    lines.append("\n## 3. Top 10 largest epoch-to-epoch **spikes** in val_hd95\n")
    lines.append("(Δ = hd95[epoch] − hd95[epoch−1]; larger positive = worse spike)\n")
    lines.append("| Rank | Epoch | Δ val_hd95 |\n|---:|---:|---:|")
    for r, (e, delta) in enumerate(hd_spikes, 1):
        lines.append(f"| {r} | {e} | {delta:.6f} |")

    # 4) Correlation train_loss vs val_dice
    r_all = pearson_r(train_loss, val_dice)
    # Late phase: second half of training
    half = n // 2
    r_late = pearson_r(train_loss[half:], val_dice[half:])
    lines.append("\n## 4. Correlation: train_loss vs val_dice\n")
    lines.append(f"- **Pearson r (all epochs):** {r_all:.6f}\n")
    lines.append(f"- **Pearson r (epochs ≥ index {half + 1}, second half):** {r_late:.6f}\n")
    lines.append(
        "- **Overfitting read:** Negative r means lower train loss tracks with higher val dice (healthy). "
        "If train loss keeps falling while val dice stalls or falls, global r can still mask late divergence — "
        "compare r on the second half.\n"
    )

    # 5) LR at inflection epochs
    idx = {e: i for i, e in enumerate(ep)}
    key_epochs = set()
    for e, _ in dice_drops:
        key_epochs.add(e)
        key_epochs.add(e - 1)
    for e, _ in hd_spikes:
        key_epochs.add(e)
        key_epochs.add(e - 1)
    key_epochs = sorted(x for x in key_epochs if x >= ep[0] and x <= ep[-1])

    lines.append("\n## 5. Learning rate at key inflection epochs\n")
    lines.append("(Union of epochs involved in top-10 dice drops and top-10 hd95 spikes, ±1 epoch)\n")
    lines.append("| Epoch | lr |\n|---:|---:|")
    for e in key_epochs:
        i = idx.get(e)
        if i is not None:
            lines.append(f"| {e} | {lr[i]:.10e} |")

    # 6) Per-class variance
    def pvar(vals):
        return statistics.pvariance(vals) if len(vals) > 1 else 0.0

    vd = {
        "rv": pvar(dice_rv),
        "myo": pvar(dice_myo),
        "lv": pvar(dice_lv),
    }
    vh = {
        "rv": pvar(hd95_rv),
        "myo": pvar(hd95_myo),
        "lv": pvar(hd95_lv),
    }
    max_d = max(vd, key=vd.get)
    max_h = max(vh, key=vh.get)
    lines.append("\n## 6. Per-class stability (population variance across all epochs)\n")
    lines.append("| Class | var(dice) | var(hd95) |\n|---|---:|---:|")
    for c in ("rv", "myo", "lv"):
        lines.append(f"| {c} | {vd[c]:.8f} | {vh[c]:.8f} |")
    lines.append(f"\n**Highest dice variance:** {max_d} ({vd[max_d]:.8f})  \n")
    lines.append(f"**Highest hd95 variance:** {max_h} ({vh[max_h]:.8f})\n")

    # 7) Last 50 epochs: MA window 10, trend
    last50_start = n - 50
    if last50_start < 9:
        lines.append("\n## 7. Last 50 epochs MA — insufficient history\n")
        ma_dice = []
        ma_hd = []
        epochs_ma = []
    else:
        epochs_ma = []
        ma_dice = []
        ma_hd = []
        for k in range(last50_start, n):
            # backward 10-epoch window ending at k
            ma_dice.append(statistics.mean(val_dice[k - 9 : k + 1]))
            ma_hd.append(statistics.mean(val_hd95[k - 9 : k + 1]))
            epochs_ma.append(ep[k])

        x = list(range(len(epochs_ma)))
        _, b_d = linear_slope(x, ma_dice)
        _, b_h = linear_slope(x, ma_hd)

        def trend_label(slope, metric_higher_better):
            if math.isnan(slope):
                return "n/a"
            thr = 1e-6
            if abs(slope) < thr:
                return "approximately flat"
            if metric_higher_better:
                return "improving" if slope > 0 else "degrading"
            return "improving" if slope < 0 else "degrading"

        td = trend_label(b_d, True)
        th = trend_label(b_h, False)

        lines.append("\n## 7. Last 50 epochs: 10-epoch moving average & trend\n")
        lines.append(f"- **Epoch range:** {epochs_ma[0]}–{epochs_ma[-1]} (n={len(epochs_ma)} MA points)\n")
        lines.append(f"- **val_dice MA:** first={ma_dice[0]:.6f}, last={ma_dice[-1]:.6f}\n")
        lines.append(f"- **Linear slope vs index (dice MA):** {b_d:.8f} → **{td}**\n")
        lines.append(f"- **val_hd95 MA:** first={ma_hd[0]:.6f}, last={ma_hd[-1]:.6f}\n")
        lines.append(f"- **Linear slope vs index (hd95 MA, lower better):** {b_h:.8f} → **{th}**\n")

    # Summary stats for report footer
    lines.append("\n---\n")
    lines.append("## [LIMITATION]\n")
    lines.append(
        "- Single run; no confidence intervals on correlations or slopes.\n"
        "- Pearson r is linear association only; phase shifts between loss and dice are not captured.\n"
        "- hd95 spikes can reflect label noise or small validation set variability.\n"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "acdc_asym_v3_c48_training_dynamics.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    print("---")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
