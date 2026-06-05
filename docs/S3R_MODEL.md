# S3R-Net

S3R-Net is the main model path for future SpecUMamba experiments.

S3R means Spectral State-Space Retention Network. The model keeps two signals:

- `F_t`: the spatial feature map at transition `t`
- `S_t`: an explicit spectral state with shape `[B, K, C_s]`

Each transition updates both:

```text
(F_t, S_t) -> (F_{t+1}, S_{t+1})
```

## Main Components

`src/models/s3r/s3r_blocks.py` promotes the tested `ssr_full` prototype as `SSRFullBlock`.

It includes:

- rFFT radial band split
- per-band log energy, log variance, and phase coherence
- retain, update, and suppress gates
- bounded residual strength `gamma`
- residual channel gate
- large-kernel geometry refinement by default
- optional torchvision deformable convolution for ablation only
- high-frequency ratio penalty and boundary/non-boundary diagnostics

`src/models/s3r/s3r_state.py` adds the explicit spectral state:

- `SpectralStateInitializer`: initializes `S_t` from FFT band summaries
- `SpectralStateTransition`: updates state with retain/update gates
- `StateGuidedModulation`: applies bounded channel modulation from state to features

`S3RTransitionBlock` composes both:

```text
state_new = SpectralStateTransition(feat, state)
feat_new = SSRFullBlock(feat)
feat_new = StateGuidedModulation(feat_new, state_new)
```

## Models

`S3RMini` is the compact replacement for the prototype MiniSSR model:

```text
input -> SignalLift -> StateInit -> S3RTransition -> ConvBlock
      -> S3RTransition -> StateGuidedModulation -> shared head
      -> segmentation head + boundary head
```

`S3RNet` is the future architecture:

```text
input
  -> SignalLift
  -> Stage 1 @ full resolution: S3RTransition x 2
  -> AntiAliasedDownsample
  -> Stage 2: S3RTransition x 2
  -> AntiAliasedDownsample
  -> Stage 3: S3RTransition x 3
  -> state-guided reconstruction
  -> segmentation head + boundary head
```

There is no U-Net-style skip concatenation by default and no repeated HRNet-style branch exchange.

## Output Contract

S3R models return:

```python
{
    "seg_logits": Tensor,       # [B, 4, H, W]
    "output": Tensor,           # alias for legacy trainers
    "boundary_logits": Tensor,  # [B, 1, H, W]
    "gate_reg": Tensor,
    "hf_ratio_penalty": Tensor,
    "state": Tensor,            # [B, K, C_s]
    "logs": dict,               # optional
}
```

The old `specmamba`, `asym_spec_mamba`, `hrnet_dcn`, and `hrnet_resnet34` files are preserved. New model names are exposed through `src/models/__init__.py`:

- `s3r`
- `s3r_mini`
- `s3r_net`

## Supervised Training

Default command:

```bash
python src/training/train_s3r_acdc.py \
  --model s3r_mini \
  --data_dir preprocessed_data/ACDC \
  --image_size 224 \
  --epochs 200 \
  --batch_size 8 \
  --input_mode 2d \
  --save_dir weights/s3r_mini_acdc
```

Smoke test:

```bash
python src/training/train_s3r_acdc.py \
  --model s3r_mini \
  --data_dir preprocessed_data/ACDC \
  --image_size 224 \
  --epochs 2 \
  --batch_size 8 \
  --max_slices 32 \
  --save_dir weights/smoke_s3r_mini
```
