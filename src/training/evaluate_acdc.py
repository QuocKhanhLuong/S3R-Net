"""
ACDC Test-Set Evaluation — loads a trained checkpoint and evaluates on the
held-out test split with full 3D volumetric metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import json
from collections import defaultdict, OrderedDict
from scipy.ndimage import distance_transform_edt, binary_erosion
from torch.utils.data import Dataset
import glob

CLASS_NAMES = {1: 'RV', 2: 'MYO', 3: 'LV'}


class ACDCDataset25D(Dataset):
    """Stack 5 consecutive slices [k-2..k+2] as input channels."""

    def __init__(self, npy_dir):
        self.vol_paths = sorted(glob.glob(os.path.join(npy_dir, 'volumes', '*.npy')))
        self.mask_paths = sorted(glob.glob(os.path.join(npy_dir, 'masks', '*.npy')))
        self._cache: OrderedDict = OrderedDict()
        self.max_cache = 20

        metadata_path = os.path.join(npy_dir, 'metadata.json')
        volume_info = None
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                volume_info = json.load(f).get('volume_info', {})

        self.vol_names: list[str] = []
        self.slice_counts: dict[int, int] = {}
        self.index_map: list[tuple[int, int]] = []
        for i, vp in enumerate(self.vol_paths):
            vid = os.path.basename(vp).replace('.npy', '')
            self.vol_names.append(vid)
            if volume_info and vid in volume_info:
                n_slices = volume_info[vid]['num_slices']
            else:
                vol = np.load(vp, mmap_mode='r')
                n_slices = vol.shape[-1] if vol.ndim == 3 else vol.shape[0]
            self.slice_counts[i] = n_slices
            for s in range(n_slices):
                self.index_map.append((i, s))

    def _load(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        vol = np.load(self.vol_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.int64)
        if vol.ndim == 3 and vol.shape[0] > vol.shape[-1]:
            vol = vol.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)
        self._cache[idx] = (vol, mask)
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return vol, mask

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.index_map[idx]
        vol, mask = self._load(vol_idx)
        n = vol.shape[0]

        def reflect(i, n):
            if i < 0:
                return -i
            if i >= n:
                return 2 * (n - 1) - i
            return i

        channels = []
        for offset in range(-2, 3):
            s = reflect(slice_idx + offset, n)
            channels.append(vol[s])

        img = np.stack(channels, axis=0)
        return (torch.from_numpy(img),
                torch.from_numpy(mask[slice_idx].copy()),
                vol_idx, slice_idx)


def evaluate_test(model, dataset, device, num_classes=4, volume_meta=None):
    model.eval()
    vol_preds = defaultdict(list)
    vol_targets = defaultdict(list)

    with torch.no_grad():
        for i in range(len(dataset)):
            img, target, vol_idx, slice_idx = dataset[i]
            img = img.unsqueeze(0).to(device)
            pred = model(img)['output'].argmax(1).squeeze(0).cpu().numpy()
            vol_preds[vol_idx].append((slice_idx, pred))
            vol_targets[vol_idx].append((slice_idx, target.numpy()))

    per_vol = {}
    dice_all = {c: [] for c in range(1, num_classes)}
    hd95_all = {c: [] for c in range(1, num_classes)}
    prec_all = {c: [] for c in range(1, num_classes)}
    recall_all = {c: [] for c in range(1, num_classes)}
    acc_all = {c: [] for c in range(1, num_classes)}

    for vol_idx in sorted(vol_preds.keys()):
        pred_3d = np.stack([p[1] for p in sorted(vol_preds[vol_idx], key=lambda x: x[0])], axis=0)
        target_3d = np.stack([t[1] for t in sorted(vol_targets[vol_idx], key=lambda x: x[0])], axis=0)

        spacing = None
        if volume_meta is not None and vol_idx in volume_meta:
            spacing = volume_meta[vol_idx].get('effective_spacing')

        vol_name = dataset.vol_names[vol_idx]
        vol_metrics = {}

        for c in range(1, num_classes):
            pred_c = (pred_3d == c)
            target_c = (target_3d == c)
            cn = CLASS_NAMES[c]

            if not target_c.any():
                if not pred_c.any():
                    continue
                dice_all[c].append(0.0); hd95_all[c].append(100.0)
                prec_all[c].append(0.0); recall_all[c].append(0.0)
                acc_all[c].append(float((~pred_c).sum() / pred_c.size))
                vol_metrics[cn] = {'dice': 0.0, 'hd95': 100.0, 'prec': 0.0, 'recall': 0.0, 'acc': 0.0}
                continue

            tp = float((pred_c & target_c).sum())
            fp = float((pred_c & ~target_c).sum())
            fn = float((~pred_c & target_c).sum())
            tn = float((~pred_c & ~target_c).sum())

            dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

            dice_all[c].append(dice)
            prec_all[c].append(prec)
            recall_all[c].append(rec)
            acc_all[c].append(acc)

            if pred_c.any() and target_c.any():
                pb = pred_c ^ binary_erosion(pred_c)
                tb = target_c ^ binary_erosion(target_c)
                if not pb.any():
                    pb = pred_c
                if not tb.any():
                    tb = target_c
                d1 = distance_transform_edt(~target_c, sampling=spacing)[pb]
                d2 = distance_transform_edt(~pred_c, sampling=spacing)[tb]
                hd95 = float(np.percentile(np.concatenate([d1, d2]), 95))
            else:
                hd95 = 100.0

            hd95_all[c].append(hd95)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            vol_metrics[cn] = {
                'dice': round(dice, 4), 'hd95': round(hd95, 4),
                'prec': round(prec, 4), 'recall': round(rec, 4),
                'acc': round(acc, 4), 'f1': round(f1, 4),
            }

        per_vol[vol_name] = vol_metrics

    sm = lambda lst: float(np.mean(lst)) if lst else 0.0
    summary = {}
    for c in range(1, num_classes):
        cn = CLASS_NAMES[c]
        d, h, p, r, a = sm(dice_all[c]), sm(hd95_all[c]), sm(prec_all[c]), sm(recall_all[c]), sm(acc_all[c])
        f1 = 2 * p * r / (p + r + 1e-8)
        summary[cn] = {
            'dice': round(d, 4), 'hd95': round(h, 4),
            'prec': round(p, 4), 'recall': round(r, 4),
            'acc': round(a, 4), 'f1': round(f1, 4),
            'n_volumes': len(dice_all[c]),
        }

    avg_dice = np.mean([summary[cn]['dice'] for cn in CLASS_NAMES.values()])
    avg_hd95 = np.mean([summary[cn]['hd95'] for cn in CLASS_NAMES.values()])
    avg_prec = np.mean([summary[cn]['prec'] for cn in CLASS_NAMES.values()])
    avg_rec = np.mean([summary[cn]['recall'] for cn in CLASS_NAMES.values()])
    avg_acc = np.mean([summary[cn]['acc'] for cn in CLASS_NAMES.values()])
    avg_f1 = np.mean([summary[cn]['f1'] for cn in CLASS_NAMES.values()])

    summary['avg_foreground'] = {
        'dice': round(float(avg_dice), 4), 'hd95': round(float(avg_hd95), 4),
        'prec': round(float(avg_prec), 4), 'recall': round(float(avg_rec), 4),
        'acc': round(float(avg_acc), 4), 'f1': round(float(avg_f1), 4),
    }

    return summary, per_vol


def main():
    parser = argparse.ArgumentParser(description='ACDC Test-Set Evaluation')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to preprocessed test data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--model', type=str, default='asym_spec_mamba',
                        choices=['specmamba', 'asym_spec_mamba', 'hrnet_dcn', 'hrnet_resnet34'])
    parser.add_argument('--base_channels', type=int, default=48)
    parser.add_argument('--hd95_unit', type=str, default='pixel',
                        choices=['pixel', 'mm'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: <checkpoint>_test_results.json)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 4

    # Load model
    if args.model == 'asym_spec_mamba':
        from models.specmamba_net import AsymSpecMambaDCN
        model = AsymSpecMambaDCN(
            in_ch=5, num_classes=num_classes,
            base_ch=args.base_channels,
        ).to(device)
        model_name = f"AsymSpecMambaDCN-C{args.base_channels}"
    elif args.model == 'specmamba':
        from models.specmamba_net import SpecMambaNet
        model = SpecMambaNet(
            in_channels=5, num_classes=num_classes,
            base_channels=args.base_channels, img_size=224,
        ).to(device)
        model_name = f"SpecMambaNet-C{args.base_channels}"
    elif args.model == 'hrnet_dcn':
        from models.hrnet_dcn import HRNetDCN
        model = HRNetDCN(
            in_channels=3, num_classes=num_classes,
            base_channels=args.base_channels,
        ).to(device)
        model_name = f"HRNetDCN-C{args.base_channels}"
    elif args.model == 'hrnet_resnet34':
        from models.hrnet_resnet34 import HRNetResNet34
        model = HRNetResNet34(
            in_channels=3, num_classes=num_classes,
            base_channels=args.base_channels,
        ).to(device)
        model_name = f"HRNetResNet34-C{args.base_channels}"

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"ACDC Test-Set Evaluation")
    print(f"{'='*60}")
    print(f"Model:      {model_name} | Params={params:,}")
    print(f"Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"Test Dir:   {args.test_dir}")
    print(f"HD95 Unit:  {args.hd95_unit}")
    print(f"Device:     {device}")

    # Load spacing metadata
    volume_meta = {}
    if args.hd95_unit == 'mm':
        metadata_path = os.path.join(args.test_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            vol_info = meta.get('volume_info', {})
            for idx, vname in enumerate(sorted(vol_info.keys())):
                if 'effective_spacing' in vol_info[vname]:
                    volume_meta[idx] = vol_info[vname]
            if volume_meta:
                sp = next(iter(volume_meta.values()))['effective_spacing']
                print(f"Spacing:    e.g. {[round(s, 2) for s in sp]}")

    # Load test dataset
    dataset = ACDCDataset25D(args.test_dir)
    print(f"Volumes:    {len(dataset.vol_paths)}")
    print(f"Slices:     {len(dataset)}")
    print(f"{'='*60}\n")

    # Evaluate
    print("Running inference...")
    summary, per_vol = evaluate_test(
        model, dataset, device,
        num_classes=num_classes, volume_meta=volume_meta if args.hd95_unit == 'mm' else None,
    )

    # Print results
    avg = summary['avg_foreground']
    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"\n--- Average Foreground ---")
    print(f"  Dice:      {avg['dice']:.4f}")
    print(f"  HD95:      {avg['hd95']:.4f} {args.hd95_unit}")
    print(f"  Precision: {avg['prec']:.4f}")
    print(f"  Recall:    {avg['recall']:.4f}")
    print(f"  Accuracy:  {avg['acc']:.4f}")
    print(f"  F1:        {avg['f1']:.4f}")

    print(f"\n--- Per-Class ---")
    print(f"  {'Class':<5}  {'Dice':>6}  {'HD95':>7}  {'Prec':>6}  {'Rec':>6}  {'Acc':>6}  {'F1':>6}  {'N':>3}")
    print(f"  {'─'*52}")
    for cn in ['RV', 'MYO', 'LV']:
        m = summary[cn]
        print(f"  {cn:<5}  {m['dice']:>6.4f}  {m['hd95']:>7.4f}  {m['prec']:>6.4f}  {m['recall']:>6.4f}  {m['acc']:>6.4f}  {m['f1']:>6.4f}  {m['n_volumes']:>3}")

    print(f"\n{'='*60}")

    # Save results
    output_path = args.output or args.checkpoint.replace('.pt', '_test_results.json')
    results = {
        'model': model_name,
        'checkpoint': os.path.basename(args.checkpoint),
        'params': params,
        'hd95_unit': args.hd95_unit,
        'num_test_volumes': len(dataset.vol_paths),
        'summary': summary,
        'per_volume': per_vol,
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_path}")


if __name__ == '__main__':
    main()
