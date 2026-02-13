# Codebase Restructure + Bug Fixes

## Goal

Reorganize 30+ scattered root files into a clean `wavessm_x/` package and fix all code review issues.

## Current Problem

```
root/
├── fass_ssm.py          # New
├── wave_ffc.py           # New 
├── losses.py             # New (overlaps with frequency_loss.py!)
├── mask_estimator.py     # New
├── model_directional_query_od.py   # Has dead PureSSM/WaveletGuidedSSM
├── main_py.py            # 1220 lines, does everything
├── frequency_loss.py     # 616 lines, overlaps losses.py
├── copyOfMain.py         # Dead file
├── odconv.py             # Dependency
├── utils.py + utils_train.py  # Scattered utils
└── 20+ more files...
```

## Proposed Structure

```
wavessm_x/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── fass_ssm.py           ← from root/fass_ssm.py (with bug fixes)
│   ├── wave_ffc.py           ← from root/wave_ffc.py (with bug fixes)
│   ├── mask_estimator.py     ← from root/mask_estimator.py (with bug fixes)
│   ├── odconv.py             ← from root/odconv.py
│   └── inpainting.py         ← extracted from model_directional_query_od.py
│                               (without dead PureSSM/WaveletGuidedSSM)
├── losses/
│   ├── __init__.py
│   ├── perceptual.py         ← HRFPL, VGGPerceptualLoss
│   ├── frequency.py          ← FrequencyLoss (merge frequency_loss.py + losses.py)
│   ├── ssim.py               ← SSIMLoss (fixed with Gaussian window)
│   ├── adversarial.py        ← AdversarialWaveletDiscriminator
│   └── combined.py           ← WaveSSMLoss, MaskAwareLoss
├── data/
│   ├── __init__.py
│   ├── dataset.py            ← from utils_train.py (OptimizedTrainDataset)
│   ├── augmentation.py       ← from utils_train.py (AdvancedAugmentation)
│   └── split.py              ← from main_py.py (get_or_create_data_split)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            ← from evaluation/metrics.py
│   └── visualize.py          ← from evaluation/visualize.py
└── utils/
    ├── __init__.py
    └── config.py             ← from utils.py, utils_train.py (parse_args, Config)

# Root level (kept)
train.py                      ← renamed from main_py.py (training loop only)
test.py                       ← existing
inference_inpaint.py          ← existing
requirements.txt              ← existing
README.md                     ← existing
```

## Bug Fixes (Applied During Restructure)

### 🔴 P0: CrossFrequencyAttention OOM

**File:** `wavessm_x/models/fass_ssm.py`

```diff
 # Switch from O(N²) spatial attention to O(C²) channel attention
-attn = torch.softmax(q.transpose(-1, -2) @ k * self.scale, dim=-1)  # (HW, HW) OOM!
-out = (v @ attn.transpose(-1, -2)).view(B, C, H, W)
+attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)  # (C, C) safe
+out = (attn @ v).view(B, C, H, W)
```

### 🟡 P1: SSIMLoss — Use Gaussian window

**File:** `wavessm_x/losses/ssim.py`

Remove unused `sigma` param, OR implement proper Gaussian-weighted SSIM using `sigma`.

### 🟡 P1: Remove dead code

**File:** `wavessm_x/models/inpainting.py`

Remove `PureSSM` (lines 151-268) and `WaveletGuidedSSM` (lines 271-512) — ~360 lines of dead code.

### 🟡 P2: Double DWT in DualStreamFASS

**File:** `wavessm_x/models/fass_ssm.py`

Add `use_freq_modulation=False` when FASS is used inside DualStreamFASS (skip internal DWT since DualStream already handles frequency separation).

### 🟢 P3: Minor fixes

- Clamp `MultiScaleWaveFFC` ratio to `max(0.1, ...)`
- Fix LPIPS zero tensor `requires_grad=False`
- Add `in_channels` adapter to MaskEstimator

## Files to Delete After Restructure

| File | Reason |
|------|--------|
| `copyOfMain.py` | Dead backup file |
| `frequency_loss.py` | Merged into `wavessm_x/losses/frequency.py` |
| `brainstorm.md` | Planning artifact, not code |
| `wavelet_guided_ssm.md` | Docs, move to paper/ |
| `filestructure.txt` | Outdated |

## Verification Plan

### Automated
```bash
cd c:\Priyansh\3rdyear\Capstone\Blind_Omni_Wav_Net
python -c "from wavessm_x.models.fass_ssm import FrequencyAdaptiveSSM, DualStreamFASS"
python -c "from wavessm_x.models.inpainting import Inpainting"
python -c "from wavessm_x.losses.combined import WaveSSMLoss"
python -c "from wavessm_x.data.dataset import OptimizedTrainDataset"
python -m py_compile wavessm_x/models/fass_ssm.py
python -m py_compile wavessm_x/models/wave_ffc.py
python -m py_compile wavessm_x/losses/combined.py
python -m py_compile train.py
```

### Smoke Test
```python
import torch
from wavessm_x.models.fass_ssm import FrequencyAdaptiveSSM, DualStreamFASS, CrossFrequencyAttention

# Test CrossFrequencyAttention doesn't OOM
cfa = CrossFrequencyAttention(64)
ll = torch.randn(1, 64, 128, 128)
hf = (torch.randn(1,64,128,128), torch.randn(1,64,128,128), torch.randn(1,64,128,128))
out = cfa(ll, hf)  # Should NOT OOM
print(f"CFA output: {out.shape}")  # (1, 64, 128, 128)
```
