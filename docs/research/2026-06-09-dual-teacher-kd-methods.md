# Research Note: Cac co che Dual-Teacher KD cho SpecUMamba

Ngay: 2026-06-09

## 1. Muc tieu va boi canh repo

Muc tieu la chon co che knowledge distillation (KD) thuc dung cho bai toan cardiac MRI segmentation trong SpecUMamba:

- Student: S3R / S3R-Mini / S3R-Net.
- Teacher A: MedSAM2, dung nhu semantic / prompt-driven field teacher.
- Teacher B: CineMA, dung nhu cardiac anatomy / boundary / spectral teacher.
- Rang buoc trien khai: teacher chi xuat hien luc train; inference cua S3R van teacher-free.

Boi canh hien tai cua repo:

- `src/losses/agreement_kd.py` da co output-level dual-teacher KD, Jensen-Shannon agreement map, region interior/boundary routing, CineMA boundary KD, fused KD, va spectral boundary KD.
- `src/training/train_s3r_acdc.py` da co `dual_teacher_kd` config, CLI flags, teacher stubs, online teacher, va cache path.
- `src/distillation/distill_losses.py` va `src/distillation/teacher_targets.py` da co y tuong cache-first, region routing, state/spectral KD cho pipeline distillation rieng.
- Diem yeu can xu ly: agreement hien tai chu yeu anh huong den fusion diagnostics; neu nhan cung mot `agreement_factor` vao ca hai teacher weight roi normalize, factor nay co the bi tri tieu trong ty le `W_M3` / `W_C`. Vi vay disagreement chua chac da lam giam ap luc `loss_fuse`.

## 2. Co so tu literature

KD goc cua Hinton et al. dat nen tang cho viec day student bang soft targets va temperature scaling, voi muc tieu dua kien thuc cua ensemble/teacher lon vao student nho hon ma khong can teacher luc inference. Survey cua Gou et al. phan loai KD theo logits, features, relations, schemes va ung dung, cho thay output KD chi la mot lop trong tap co che rong hon.

Voi multi-teacher KD, rui ro lon nhat la naive averaging: neu mot teacher sai hoac over-confident, student se hoc pseudo-target sai. CA-MKD de xuat gan reliability weight theo teacher prediction quality; dieu nay phu hop voi boi canh MedSAM2 va CineMA co domain strength khac nhau. Voi dense prediction/segmentation, cac paper nhu BPKD va KD cho medical image segmentation nhan manh rang body/interior va edge/boundary can tin hieu distillation khac nhau, thay vi doi xu moi pixel nhu nhau. FitNets, TransKD va GKD tu vision foundation models cho thay feature/intermediate KD co ich, nhung doi hoi projection/feature taps ro rang nen nen lam sau output/boundary KD.

Rieng teacher trong repo:

- MedSAM2 la prompt-driven medical segmentation foundation model duoc fine-tune tren nhieu medical datasets/modality, phu hop vai tro semantic field teacher nhung can canh giac prompt/GT-box leakage khi danh gia.
- CineMA la cine cardiac MRI foundation model, co model weights ACDC SAX segmentation tren Hugging Face, phu hop vai tro cardiac anatomy va boundary teacher.

## 3. Danh sach phuong phap va so sanh co che

| Phuong phap | Co che | Tin vao dieu gi | Uu diem | Rui ro | Phu hop voi repo |
|---|---|---|---|---|---|
| 1. Simple weighted sum | Tong loss co dinh: `L = L_seg + a*KD_M3 + b*KD_CineMA + c*KD_fuse + d*KD_spec`. | Tin vao lambda global. | De cai, de ablation, on dinh. | Khong biet vung nao teacher gioi/yeu; khong xu ly disagreement. | Nen giu lam baseline. Repo da co `lambda_field`, `lambda_cine_boundary`, `lambda_fuse`, `lambda_spec`. |
| 2. Probability/logit averaging | Trung binh `P_M3` va `P_C`, hoac weighted average theo constant, roi KL student vao target fused. | Tin rang ensemble average tot hon tung teacher. | Don gian, dung logic KD ensemble goc. | Che dau xung dot; mot teacher sai co the keo fused target sai. | Khong nen lam co che chinh cho MedSAM2 + CineMA vi hai teacher co role khac nhau. |
| 3. Agreement-aware fusion | Tinh agreement theo `A = exp(-JS(P_M3, P_C))`; dung agreement trong fusion/weighting. | Teacher dong thuan thi pseudo-target dang tin hon. | Phu hop dual-teacher, co diagnostic tot. | Neu agreement chi nhan vao ca hai teacher weight truoc normalize, no co the khong giam loss o vung bat dong. | Da co mot phan. Nen sua theo huong agreement gate truc tiep `loss_fuse`. |
| 4. Uncertainty/confidence weighting | Dung entropy/confidence map `C_M3`, `C_C`, calibration, hoac do gan voi GT trong train de gan reliability. | Teacher tu tin va/hoac gan GT thi dang tin hon. | Giam tac dong teacher uncertain; phu hop CA-MKD. | Confidence co the miscalibrated; over-confident wrong prediction van nguy hiem. | Nen dung ket hop voi agreement, khong dung mot minh. Repo da co `C_M3` va `C_C`. |
| 5. Region/boundary routing | MedSAM2 supervise interior/semantic field; CineMA supervise boundary/anatomy. `W_interior = 1 - W_boundary`, `W_boundary` tu GT boundary band hoac CineMA boundary. | Moi teacher co chuyen mon theo region. | Rat hop cardiac segmentation: MYO/LV/RV loi nhieu o edge va contour. | Can boundary map dang tin; GT-derived boundary chi duoc dung luc train. | Rat phu hop. Repo da co `W_interior`, `W_boundary`, `cinema_boundary_kd_loss`. |
| 6. Curriculum/schedule | Epoch dau train supervised hoac single-teacher; sau do ramp `lambda_field`, `lambda_cine_boundary`, `lambda_fuse`, `lambda_spec`. | Student can hoc anatomy co ban truoc khi nghe teacher phuc tap. | Giam noisy KD o dau training; de kiem soat. | Them schedule plumbing, can nhieu ablation. | Nen lam sau selective gate. |
| 7. Cache-first/offline distillation | Precompute `P_M3`, `C_M3`, `P_C`, `C_C`, `B_C` va metadata; train tu cache. | Teacher outputs da duoc dong bang va kiem soat provenance. | Reproducible, tranh online OOM, hop long runs. | Cache drift theo split, image size, class order, checkpoint hash. | Bat buoc cho real dual-teacher experiment dai. Repo da co cache path. |
| 8. Selective KD / disagreement gating | Downweight hoac skip fused KD khi teachers bat dong; supervised CE/Dice giu vai tro anchor. | Disagreement la tin hieu rui ro pseudo-label. | Tranh ep student hoc target trung binh sai; thay doi nho. | Threshold/power can tune; gate qua manh co the bo qua minority teacher dung. | Khuyen nghi implement ngay. No fix dung diem yeu hien tai cua `loss_fuse`. |
| 9. Disagreement regularization | Them loss phat khi student qua tu tin o vung teachers bat dong, hoac khuyen khich student gan GT hon teacher o conflict region. | Conflict region nen hoc can than, khong nen confident theo fused pseudo-label. | Co the tang robustness va calibration. | De lam phuc tap loss; neu khong co GT/uncertainty ro co the gay underfit boundary. | Nen ghi log va ablate sau selective gate. |
| 10. Feature/state/spectral KD | Distill intermediate features, S3R state bands, boundary spectra, patch embeddings. | Teacher co structural knowledge khong nam het trong logits. | Hop voi S3R spectral/state design; co the tang boundary quality. | Can feature taps/projection heads/cache moi; implementation risk cao hon. | Spectral boundary KD da nen giu. Feature/state KD nen phase sau. |
| 11. Multi-seed CineMA ensemble | Dung nhieu CineMA seeds `acdc_sax_0/1/2` lam anatomy teacher ensemble. | Ensemble CineMA co the giam variance cua mot checkpoint. | Tang chat luong boundary teacher neu compute cho phep. | Tang storage/compute; can cache de kha thi. | Khong lam ngay; chi nen dung cho offline cache experiment. |

## 4. Khuyen nghi implement ngay

Nen implement **selective agreement-gated fused KD**.

Y tuong: giu nguyen field KD, CineMA boundary KD va spectral boundary KD; chi them weight map cho `loss_fuse` dua tren agreement map san co.

Cong thuc de xuat:

```text
agreement = exp(-JS(P_M3, P_C))
disagreement = 1 - agreement
W_fuse = min_weight + (1 - min_weight) * agreement^power
loss_fuse = KL(student_logits, P_F, weight=W_fuse)
```

Config de xuat:

```yaml
dual_teacher_kd:
  fused_kd_weight_mode: agreement   # none | agreement
  fused_kd_min_weight: 0.10
  fused_kd_agreement_power: 1.0
```

Ly do nen chon co che nay truoc:

- Incremental: thay doi chinh nam trong `src/losses/agreement_kd.py` va config parsing trong `src/training/train_s3r_acdc.py`.
- Khong doi teacher wrappers, dataset, cache schema bat buoc, hay inference path.
- Bien agreement tu diagnostic thanh tin hieu training thuc su.
- Tuong thich voi single-teacher fallback: khi MedSAM2 missing va repo fallback `p_m3 = p_c`, agreement se cao, nen gate khong vo tinh tat KD.
- Co ablation ro: `fused_kd_weight_mode=none` vs `agreement`.

Log nen them:

- `teacher_disagreement_mean = 1 - agreement_mean`
- `fuse_weight_mean`
- `fuse_weight_min`
- `fuse_weight_max`

## 5. Ablation toi thieu

1. Baseline S3R, no KD.
2. Current fixed dual-teacher KD.
3. Region-routed KD without fused KD.
4. Agreement-gated fused KD.
5. Agreement-gated fused KD + spectral boundary KD.
6. Cache-first agreement-gated KD voi real CineMA va MedSAM2 outputs.

Metrics nen doc cung nhau:

- Dice foreground va Dice tung class RV/MYO/LV.
- Boundary F1, surface Dice, HD95/ASSD neu co.
- KD diagnostics: `loss_field`, `loss_cine_boundary`, `loss_fuse`, `loss_spec`, `loss_kd`, `agreement_mean`, `teacher_disagreement_mean`, `fuse_weight_mean`, `W_M3_mean`, `W_C_mean`.
- S3R diagnostics hien co: `high_freq_ratio`, `boundary_to_nonboundary_high_ratio`, `hf_ratio_penalty`, `gamma`, `update_gate_mean`, `suppress_gate_mean`.

## 6. Handoff cho ATeam implementation

Pham vi implement de xuat:

- Them config keys:
  - `dual_teacher_kd.fused_kd_weight_mode`
  - `dual_teacher_kd.fused_kd_min_weight`
  - `dual_teacher_kd.fused_kd_agreement_power`
- Sua `fused_kd_loss` de nhan optional `weight`.
- Trong `compute_dual_teacher_kd_loss`, neu mode la `agreement`, tinh `W_fuse` tu `fusion["agreement"]` va truyen vao fused KD.
- Log `teacher_disagreement_mean`, `fuse_weight_mean`, `fuse_weight_min`, `fuse_weight_max`.
- Them tests cho:
  - loss finite voi mode `none` va `agreement`;
  - `W_fuse` nam trong `[min_weight, 1]`;
  - lower agreement lam lower weighted fused KD contribution;
  - default behavior van gan voi current behavior khi mode `none`.
- Update `docs/DUAL_TEACHER_KD.md` va optional smoke config trong `configs/`.

Acceptance criteria:

- Default khong doi behavior: unweighted fused KD.
- Opt-in config/CLI bat duoc agreement-gated fused KD.
- Stub CPU smoke van chay khong can real teacher weights.
- Khong lam thay doi inference path cua S3R.

## 7. Ket luan

Trong boi canh SpecUMamba, dual-teacher KD khong nen la "average hai teacher". Co che hop ly nhat la xem MedSAM2 va CineMA nhu hai nguon chuyen mon: MedSAM2 cho semantic field/interior, CineMA cho anatomy/boundary, va teacher agreement/confidence la reliability gate. Buoc implement tot nhat luc nay la **agreement-gated fused KD** vi nho, do duoc, tuong thich code hien co, va truc tiep giam rui ro hoc pseudo-target sai o vung teachers bat dong.

## 8. Nguon tham khao

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. https://arxiv.org/abs/1503.02531
- Gou, J., Yu, B., Maybank, S. J., & Tao, D. (2021). Knowledge Distillation: A Survey. https://arxiv.org/abs/2006.05525
- Zhang, H., Chen, D., & Wang, C. (2022). Confidence-Aware Multi-Teacher Knowledge Distillation. https://arxiv.org/abs/2201.00007
- Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2015). FitNets: Hints for Thin Deep Nets. https://arxiv.org/abs/1412.6550
- Qin, D., Bu, J., Liu, Z., Shen, X., Zhou, S., Gu, J., Wang, Z., Wu, L., & Dai, H. (2021). Efficient Medical Image Segmentation Based on Knowledge Distillation. https://arxiv.org/abs/2108.09987
- Liu, L., Wang, Z., Phan, M. H., Zhang, B., Ge, J., & Liu, Y. (2023). BPKD: Boundary Privileged Knowledge Distillation For Semantic Segmentation. https://arxiv.org/abs/2306.08075
- Liu, R., Yang, K., Roitberg, A., Zhang, J., Peng, K., Liu, H., Wang, Y., & Stiefelhagen, R. (2024). TransKD: Transformer Knowledge Distillation for Efficient Semantic Segmentation. https://arxiv.org/abs/2202.13393
- Yang, G., Fini, E., Xu, D., Rota, P., Ding, M., Nabi, M., Alameda-Pineda, X., & Ricci, E. (2022). Uncertainty-aware Contrastive Distillation for Incremental Semantic Segmentation. https://arxiv.org/abs/2203.14098
- Karri, M., Arya, A. S., Biswas, K., Gennaro, N., Cicek, V., Durak, G., Velichko, Y. S., & Bagci, U. (2025). Uncertainty-Guided Cross Attention Ensemble Mean Teacher for Semi-supervised Medical Image Segmentation. https://arxiv.org/abs/2412.15380
- Lv, C., Zhao, D., Wang, S., Quan, D., Huyan, N., Sebe, N., & Zhong, Z. (2026). Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation. https://arxiv.org/abs/2603.02554
- Fu, Y., Bai, W., Yi, W., Manisty, C., Bhuva, A. N., Treibel, T. A., Moon, J. C., Clarkson, M. J., Davies, R. H., & Hu, Y. (2025). A versatile foundation model for cine cardiac magnetic resonance image analysis tasks. https://arxiv.org/abs/2506.00679
- CineMA GitHub repository. https://github.com/mathpluscode/CineMA
- CineMA Hugging Face model card and ACDC SAX checkpoints. https://huggingface.co/mathpluscode/CineMA
- Ma, J., Yang, Z., Kim, S., Chen, B., Baharoon, M., Fallahpour, A., Asakereh, R., Lyu, H., & Wang, B. (2025). MedSAM2: Segment Anything in 3D Medical Images and Videos. https://arxiv.org/abs/2504.03600
- MedSAM2 GitHub repository. https://github.com/bowang-lab/MedSAM2
- MedSAM2 Hugging Face model card and checkpoints. https://huggingface.co/wanglab/MedSAM2
