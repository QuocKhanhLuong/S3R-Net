# ACDC asym v3 c48 — training dynamics

**Source:** `/home/linhdang/workspace/quockhanh_workspace/SpecMamba/weights/acdc_asym_v3_c48_history.json`  
**Epochs:** 153 (epoch 1–153)


## 1. Oscillation: std dev in 10-epoch windows

| Window (epochs) | std(val_dice) | std(val_hd95) |
|---|---|---|
| 1–10 | 0.158198 | 25.298822 |
| 11–20 | 0.023286 | 1.923341 |
| 21–30 | 0.015220 | 2.610404 |
| 31–40 | 0.007681 | 1.338586 |
| 41–50 | 0.010233 | 1.092215 |
| 51–60 | 0.008414 | 0.686037 |
| 61–70 | 0.003277 | 0.589782 |
| 71–80 | 0.003067 | 0.405460 |
| 81–90 | 0.002822 | 0.330210 |
| 91–100 | 0.002116 | 0.333347 |
| 101–110 | 0.001690 | 0.090750 |
| 111–120 | 0.001842 | 0.036672 |
| 121–130 | 0.001023 | 0.051828 |
| 131–140 | 0.000634 | 0.026820 |
| 141–150 | 0.000445 | 0.015326 |
| 151–153 | 0.000271 | 0.012564 |

*Population std (pstdev) within each window.*


## 2. Top 10 largest epoch-to-epoch **drops** in val_dice

(Δ = dice[epoch] − dice[epoch−1]; more negative = larger drop)

| Rank | Epoch | Δ val_dice |
|---:|---:|---:|
| 1 | 10 | -0.041495 |
| 2 | 24 | -0.033340 |
| 3 | 41 | -0.019169 |
| 4 | 48 | -0.016923 |
| 5 | 15 | -0.015833 |
| 6 | 46 | -0.014503 |
| 7 | 33 | -0.012413 |
| 8 | 40 | -0.012094 |
| 9 | 82 | -0.009758 |
| 10 | 27 | -0.009622 |

## 3. Top 10 largest epoch-to-epoch **spikes** in val_hd95

(Δ = hd95[epoch] − hd95[epoch−1]; larger positive = worse spike)

| Rank | Epoch | Δ val_hd95 |
|---:|---:|---:|
| 1 | 5 | 11.224393 |
| 2 | 24 | 7.411956 |
| 3 | 15 | 5.386180 |
| 4 | 33 | 3.526017 |
| 5 | 20 | 2.818807 |
| 6 | 48 | 2.677142 |
| 7 | 41 | 2.269709 |
| 8 | 12 | 2.173074 |
| 9 | 63 | 1.860951 |
| 10 | 51 | 1.838499 |

## 4. Correlation: train_loss vs val_dice

- **Pearson r (all epochs):** -0.958969

- **Pearson r (epochs ≥ index 77, second half):** -0.582577

- **Overfitting read:** Negative r means lower train loss tracks with higher val dice (healthy). If train loss keeps falling while val dice stalls or falls, global r can still mask late divergence — compare r on the second half.


## 5. Learning rate at key inflection epochs

(Union of epochs involved in top-10 dice drops and top-10 hd95 spikes, ±1 epoch)

| Epoch | lr |
|---:|---:|
| 4 | 6.0000000000e-05 |
| 5 | 7.5000000000e-05 |
| 9 | 1.3500000000e-04 |
| 10 | 1.5000000000e-04 |
| 11 | 1.6500000000e-04 |
| 12 | 1.8000000000e-04 |
| 14 | 2.1000000000e-04 |
| 15 | 2.2500000000e-04 |
| 19 | 2.8500000000e-04 |
| 20 | 3.0000000000e-04 |
| 23 | 2.9987412419e-04 |
| 24 | 2.9977624513e-04 |
| 26 | 2.9949670808e-04 |
| 27 | 2.9931510224e-04 |
| 32 | 2.9799021083e-04 |
| 33 | 2.9764220618e-04 |
| 39 | 2.9497855367e-04 |
| 40 | 2.9443944724e-04 |
| 41 | 2.9387340235e-04 |
| 45 | 2.9134202527e-04 |
| 46 | 2.9064297198e-04 |
| 47 | 2.8991768852e-04 |
| 48 | 2.8916631022e-04 |
| 50 | 2.8758583466e-04 |
| 51 | 2.8675703227e-04 |
| 62 | 1.3799839345e-04 |
| 63 | 1.3743693164e-04 |
| 81 | 1.2545193317e-04 |
| 82 | 1.2468960262e-04 |

## 6. Per-class stability (population variance across all epochs)

| Class | var(dice) | var(hd95) |
|---|---:|---:|
| rv | 0.01196213 | 200.47501653 |
| myo | 0.00744871 | 100.27243174 |
| lv | 0.00630741 | 87.03352339 |

**Highest dice variance:** rv (0.01196213)  

**Highest hd95 variance:** rv (200.47501653)


## 7. Last 50 epochs: 10-epoch moving average & trend

- **Epoch range:** 104–153 (n=50 MA points)

- **val_dice MA:** first=0.902579, last=0.903016

- **Linear slope vs index (dice MA):** 0.00002607 → **improving**

- **val_hd95 MA:** first=1.206880, last=1.192756

- **Linear slope vs index (hd95 MA, lower better):** -0.00128102 → **improving**


---

## [LIMITATION]

- Single run; no confidence intervals on correlations or slopes.
- Pearson r is linear association only; phase shifts between loss and dice are not captured.
- hd95 spikes can reflect label noise or small validation set variability.
