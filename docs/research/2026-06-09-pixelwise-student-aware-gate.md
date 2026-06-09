# Research Note: Pixel-wise Student-aware Reliability Gate

Date: 2026-06-09

Scope: theory and experiment plan for a lightweight CNN agreement/reliability
gate in SpecUMamba dual-teacher KD. The target setup is:

- Student: S3R / SpecMamba student, already strong through spectral and state-memory priors.
- Teacher 1: CineMA as semantic/ROI filling teacher for DSC and class consistency.
- Teacher 2: MedSAM2 as boundary/shape teacher for HD95, ASSD, Boundary F1, and surface quality.
- Gate: pixel-wise, student-aware reliability maps that decide where KD should
  follow CineMA, where it should follow MedSAM2, and where teacher KD should be
  suppressed so supervised GT loss remains the anchor.

The key premise is not that two teachers should beat a stronger batch-size-8
single-teacher setup by default. The goal is to maximize the amount of usable
image-space knowledge transferred into a strong S3R student without changing the
student backbone or using teacher ensembles at inference.

## Theory

The existing dual-teacher path uses analytic agreement:

```text
A(x,y) = exp(-JS(P_MedSAM2(x,y), P_CineMA(x,y)))
```

This is a sensible first gate, but the training log shows agreement saturating
near `0.996`. When the agreement map is almost always close to one, it cannot
separate three cases that matter in cardiac MRI:

1. both teachers are correct and should be distilled;
2. one teacher is locally more useful than the other;
3. both teachers conflict or are noisy, so supervised CE/Dice should dominate.

A learned spatial reliability gate estimates this routing explicitly. It should
not become a third segmenter. It should only consume teacher outputs and detached
student uncertainty, then emit reliability weights:

```text
G_theta(input_maps) -> W_sem, W_char, W_ignore

W_sem(x,y)    : reliability of CineMA semantic / ROI filling KD
W_char(x,y)   : reliability of MedSAM2 boundary / characteristic KD
W_ignore(x,y) : region where teacher KD should be downweighted
```

The gate is student-aware because a strong S3R student does not need the same
teacher signal at every pixel. A pixel where the student is already confident and
teachers disagree should not receive the same KD pressure as a pixel where the
student is uncertain and one teacher has a clear local prior. Student signals
must be detached:

```python
student_probs_detached = student_probs.detach()
```

Recommended gate inputs:

- `P_C`: CineMA 4-class probabilities.
- `FG_C`: CineMA foreground probability.
- `H_C` / `Conf_C`: CineMA entropy or confidence.
- `P_M3` or `FG_M3`: MedSAM2 foreground/class probability.
- `B_M3`: MedSAM2 boundary map.
- optional `D_M3`: MedSAM2 distance map.
- `|FG_C - FG_M3|`: teacher foreground disagreement.
- `H_S`: detached student entropy.
- `Margin_S` / `Conf_S`: detached student margin or confidence.
- `B_S`: detached student boundary probability.

Recommended architecture:

```text
Conv3x3(C_in -> 32) + GroupNorm + GELU
Conv3x3(32 -> 32) + GroupNorm + GELU
Conv3x3(32 -> 16) + GroupNorm + GELU
Conv1x1(16 -> 3)
softmax over [W_sem, W_char, W_ignore]
```

Use a warmup schedule:

- epochs `0..E_warm`: gate uses teacher-only features; student uncertainty
  channels are zeroed.
- after warmup: enable detached student entropy, margin, and boundary signals.

The KD objective becomes:

```text
P_gate = normalize(W_sem * P_CineMA + W_char * P_MedSAM2)

L_sem    = KL(student, P_CineMA,  weight=W_sem)
L_char   = KL(student, P_MedSAM2, weight=W_char)
L_fuse   = KL(student, P_gate,    weight=1 - W_ignore)
L_gate   = L_seg
         + lambda_sem  * L_sem
         + lambda_char * L_char
         + lambda_fuse * L_fuse
         + lambda_spec * L_spec
         + gate_regularizers
```

Gate regularizers should be small and explicit:

- entropy or balance regularizer to prevent one-teacher collapse;
- total-variation/smoothness penalty to prevent speckled gate maps;
- optional early analytic-prior loss that decays after warmup;
- sparsity penalty on `W_ignore` so the gate cannot hide all hard pixels.

Implementation guardrail: the gate must be an `nn.Module` registered with the
training path so its parameters enter the optimizer, checkpoint, AMP/DDP, and
resume logic. It should not be hidden inside a stateless loss helper.

## 1. Research Question

Can a lightweight pixel-wise, student-aware CNN reliability gate improve
dual-teacher distillation for cardiac MRI segmentation by routing CineMA
semantic/ROI knowledge and MedSAM2 boundary/shape knowledge, thereby maximizing
DSC while preserving or improving HD95 and ASSD, without changing the S3R
backbone or using an inference-time teacher ensemble?

Sub-questions:

- Does a learned pixel-wise gate outperform the current analytic agreement gate
  when teacher agreement is saturated?
- Does detached student uncertainty improve teacher routing beyond teacher-only
  confidence/disagreement maps?
- Can `W_ignore` reduce negative transfer in disagreement regions without
  suppressing useful teacher knowledge?
- Is the role assignment `CineMA -> semantic/ROI` and
  `MedSAM2 -> boundary/shape` empirically stronger than the reverse assignment?

## 2. Hypotheses

H1. Pixel-wise reliability routing improves mean DSC over fixed or analytic
agreement-based dual-teacher KD because the gate can assign different teachers
to interior, boundary, and conflict pixels in the same image.

H2. Student-aware routing improves over teacher-only routing after warmup because
student entropy and margin identify where the strong S3R student still needs
external image-space knowledge.

H3. CineMA-weighted semantic KD improves ROI filling and class consistency,
especially for RV/MYO/LV foreground coverage, and should mainly lift DSC and
recall.

H4. MedSAM2-weighted boundary/characteristic KD improves boundary and surface
metrics, especially HD95, ASSD, Boundary F1, and surface Dice, without needing
MedSAM2 at inference.

H5. `W_ignore` improves robustness in teacher-disagreement regions by reducing
KD pressure and allowing supervised GT loss to decide, but it must be regularized
so it does not become a shortcut for avoiding hard pixels.

H6. A residual or prior-initialized learned gate should be more stable than a
free CNN gate from epoch 1, because it starts from the current analytic policy
and only learns corrections.

## 3. Ablation Matrix

| ID | Setting | Gate inputs | Purpose | Expected evidence |
|---|---|---|---|---|
| A0 | S3R supervised baseline | none | Establish student-only ceiling/floor | Strong baseline; no teacher leakage risk |
| A1 | CineMA-only KD | `P_C`, `C_C`, optional `B_C` | Measure semantic/ROI teacher value | DSC and recall improve if CineMA fills anatomy well |
| A2 | MedSAM2-only KD | `P_M3`, `C_M3`, `B_M3` | Measure boundary/shape teacher value | HD95/ASSD/Boundary F1 improve if MedSAM2 contours help |
| A3 | Fixed dual-teacher KD | fixed lambdas | Check naive combined knowledge | May help but can average conflicting targets |
| A4 | Current analytic agreement gate | `A=exp(-JS)` | Baseline for current implementation | If agreement mean stays near 0.996, routing remains weak |
| A5 | CNN gate, teacher-only | teacher probs/confidence/boundary/disagreement | Test learned dense reliability without student feedback | Should beat A4 if non-saturated features are useful |
| A6 | CNN gate + student uncertainty, no warmup | A5 + `H_S`, `Margin_S`, `B_S` | Test risk of early student-noise feedback | Could be unstable or confirm early mistakes |
| A7 | CNN gate + student uncertainty + warmup | A5, then add detached student signals | Main proposed method | Best DSC with protected HD95/ASSD |
| A8 | Residual CNN gate over analytic prior | A7 + analytic prior | Stability check | Similar or better than A7 with lower collapse risk |
| A9 | A8 + `W_ignore` | A8 + ignore head | Conflict suppression check | Lower negative transfer in disagreement pixels |
| A10 | Role swap | CineMA boundary / MedSAM2 semantic | Validate teacher-role assumption | Should underperform if chosen roles are correct |

Required metrics:

- DSC per class: RV, MYO, LV.
- Mean foreground DSC.
- HD95 and ASSD.
- Boundary F1, normalized surface Dice, or surface Dice if available.
- Precision and recall to detect ROI overfill/underfill.
- Gate diagnostics: `W_sem_mean`, `W_char_mean`, `W_ignore_mean`,
  gate entropy, means on boundary/interior/disagreement pixels.
- Teacher diagnostics: agreement histogram, not only agreement mean.
- Efficiency: additional parameters, GFLOPs, and training-time overhead.

## 4. Expected Outcomes

Expected successful pattern:

- A7/A8/A9 improve DSC over A4 without increasing HD95/ASSD.
- `W_sem` is high in confident foreground/interior regions where CineMA provides
  clean ROI filling.
- `W_char` is high on boundary bands, thin MYO contours, RV/LV edges, and
  high-surface-error regions.
- `W_ignore` is sparse and concentrated on teacher-disagreement or high-student
  uncertainty regions.
- The learned gate does not dominate model complexity; overhead should remain
  negligible compared with S3R and teachers remain train-only.

Warning signs:

- `W_sem` or `W_char` collapses to nearly one everywhere.
- `W_ignore` grows steadily and KD contribution disappears.
- DSC increases but HD95/ASSD degrade, meaning boundary teacher knowledge is not
  being used.
- Boundary metrics improve but MYO/LV/RV DSC drops, meaning the model is
  over-prioritizing contours over full label filling.
- Improvements disappear when the cache/provenance and prompt leakage checks are
  enforced.

## 5. Reviewer-risk Checklist

- Novelty risk: adaptive multi-teacher weighting, uncertainty-aware KD, and
  attention gates already exist. The claim must be dense, student-aware,
  cardiac-specific reliability routing, not generic learned teacher weighting.
- Leakage risk: MedSAM2 `gt_box` prompts and teacher caches must be generated
  only for training data used in distillation. Validation/test labels must not
  enter teacher prompts, cache generation, tuning, or gate targets.
- Teacher-role risk: literature alone does not prove CineMA is the better
  semantic teacher or MedSAM2 the better boundary teacher for ACDC. Include
  teacher-only metrics and role-swap ablation.
- Saturated-agreement risk: if `A=exp(-JS)` remains near one, do not overclaim
  analytic agreement. Show histograms and explain why entropy/margin/student
  uncertainty add information.
- Strong-student risk: S3R may already encode spectral/boundary priors. The
  method must show it transfers image-space teacher knowledge that S3R does not
  already learn from supervised loss.
- Capacity risk: reviewers may say gains come from an extra CNN. Report gate
  params/GFLOPs and compare with analytic agreement under identical training.
- Collapse risk: gate may select one teacher everywhere. Report per-region gate
  maps and histograms.
- Metric risk: optimizing DSC alone can hide worse surfaces. Report HD95, ASSD,
  and boundary/surface metrics.
- Reproducibility risk: report teacher checkpoint hash/revision, cache generation
  date, prompt mode, class mapping, image size, split IDs, and random seeds.
- Inference risk: state clearly that teachers and gate are train-time only unless
  the final implementation deliberately keeps the gate in training loss. The
  deployed S3R student remains teacher-free.

## Implementation Plan

1. Add `StudentAwareReliabilityGate(nn.Module)` as an opt-in module.
2. Add config block:

```yaml
dual_teacher_kd:
  learned_gate:
    enabled: false
    hidden_channels: 32
    use_student_uncertainty: true
    student_uncertainty_warmup_epochs: 10
    use_ignore: true
    lambda_gate_prior: 0.05
    gate_prior_decay_epochs: 30
    lambda_ignore_sparsity: 0.01
    lambda_gate_tv: 0.001
```

3. Keep cache schema unchanged: consume existing `P_M3`, `C_M3`, `P_C`, `C_C`,
   `B_C`, plus student logits available during training.
4. Add optimizer/checkpoint/resume handling for gate parameters.
5. Add tests:
   - shape, bounds, and `W_sem + W_char + W_ignore = 1`;
   - disabled fallback reproduces current analytic gate behavior;
   - student uncertainty is zeroed during warmup;
   - teacher tensors and student uncertainty are detached;
   - gate gradients are nonzero when enabled;
   - gate params are present in optimizer state and checkpoint;
   - synthetic disagreement increases `W_ignore` or reduces KD weight.
6. Add startup table rows and W&B logs for learned-gate status and gate metrics.

## Sources

- Hinton et al., "Distilling the Knowledge in a Neural Network": https://arxiv.org/abs/1503.02531
- Gou et al., "Knowledge Distillation: A Survey": https://arxiv.org/abs/2006.05525
- AMTML-KD: https://arxiv.org/abs/2103.04062
- CA-MKD: https://arxiv.org/abs/2201.00007
- Adaptive Multi-Teacher KD with Meta-Learning: https://arxiv.org/abs/2306.06634
- UA-MT uncertainty-aware medical segmentation: https://arxiv.org/abs/1907.07034
- Student Customized Knowledge Distillation: https://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Student_Customized_Knowledge_Distillation_Bridging_the_Gap_Between_Student_and_ICCV_2021_paper.html
- Structured KD for semantic segmentation: https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.html
- BPKD boundary privileged KD: https://bpkd.vmv.re/
- Boundary loss for medical image segmentation: https://pubmed.ncbi.nlm.nih.gov/33080507/
- CineMA: https://arxiv.org/abs/2506.00679
- CineMA Hugging Face: https://huggingface.co/mathpluscode/CineMA
- MedSAM2: https://arxiv.org/abs/2504.03600
- MedSAM2 GitHub: https://github.com/bowang-lab/MedSAM2
