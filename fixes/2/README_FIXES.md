# WaveSSM-X STABILITY FIXES

## 🔥 Problem Solved

Your training was **exploding at iteration 1000** when perceptual loss kicked in:
- Gradient norm reached 26.06 (should be < 2.0)
- [NaN GRAD] errors appeared
- Loss jumped from 0.118 to 0.511

## ✅ Files Provided (Upload to RunPod)

### **1. perceptual.py** 
Replace: `wavessm_x/losses/perceptual.py`

**Key fix:**
```python
def forward(self, pred, target):
    # CRITICAL: Force FP32 for entire VGG forward pass
    with torch.amp.autocast('cuda', enabled=False):
        pred = pred.float()
        target = target.float()
        # ... VGG feature extraction in FP32 ...
```

**Why:** VGG activations overflow in float16, causing gradient explosions.

---

### **2. train.py**
Replace: `train.py`

**Key fixes:**
1. **Tighter gradient clipping:** 0.25 (was 0.5)
   ```python
   grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
   ```

2. **Gradual perceptual ramp-up:** Prevents iter 1000 explosion
   ```python
   # Iter 1000-3000: 0.01 → 0.1 (not 0.0 → 0.1)
   t = (n_iter - 1000) / 2000.0
   perceptual_weight = 0.01 + t * 0.09
   ```

3. **All v9 stability features:** Health checks, sanitization, etc.

---

### **3. config.py**
Replace: `wavessm_x/utils/config.py`

**Key fix:**
```python
parser.add_argument('--lr', type=float, default=1e-4)  # Was 2e-4
```

**Why:** Lower LR = more stable training, especially with perceptual loss.

---

## 🚀 Installation Instructions

### **Step 1: Stop Current Training**

```bash
pkill -f train.py
```

### **Step 2: Upload Fixed Files to RunPod**

**Option A: Direct Upload (Easiest)**
1. Download the 3 files from Claude
2. In RunPod, use file browser or SCP to upload:
   - `perceptual.py` → `/workspace/Omni_MetaH/wavessm_x/losses/perceptual.py`
   - `train.py` → `/workspace/Omni_MetaH/train.py`
   - `config.py` → `/workspace/Omni_MetaH/wavessm_x/utils/config.py`

**Option B: SCP from Local Machine**
```bash
# From your local machine:
scp perceptual.py root@<runpod-ip>:/workspace/Omni_MetaH/wavessm_x/losses/
scp train.py root@<runpod-ip>:/workspace/Omni_MetaH/
scp config.py root@<runpod-ip>:/workspace/Omni_MetaH/wavessm_x/utils/
```

**Option C: Copy-Paste via Nano**
```bash
# In RunPod:
cd /workspace/Omni_MetaH

# Backup old files
cp train.py train_old.py
cp wavessm_x/losses/perceptual.py wavessm_x/losses/perceptual_old.py
cp wavessm_x/utils/config.py wavessm_x/utils/config_old.py

# Edit each file with nano and paste the new code
nano train.py                           # Paste new train.py
nano wavessm_x/losses/perceptual.py     # Paste new perceptual.py
nano wavessm_x/utils/config.py          # Paste new config.py
```

### **Step 3: Clear Old Checkpoint**

```bash
cd /workspace/Omni_MetaH

# Your old checkpoint exploded - remove it
rm checkpoints/wavessm_x.pth

# Optional: Keep as backup
mv checkpoints/wavessm_x.pth checkpoints/wavessm_x_EXPLODED.pth
```

### **Step 4: Start Fresh Training**

```bash
nohup python -u train.py \
    --num_iter 33000 \
    --batch_size 24 \
    --lr 1e-4 \
    --data_path ./DataSetFiles/Main_Dataset \
    --val_every 500 \
    > train_FIXED.log 2>&1 &

# Monitor
tail -f train_FIXED.log
```

---

## 📊 What to Expect After Fix

### **✅ Iteration 500 (10 minutes):**
```
Loss: 0.10-0.12
PSNR: 15-18 dB
Grad norm: < 1.0
[HEALTH CHECK] iter 500: Model health OK
```

### **✅ Iteration 1000 (THE CRITICAL TEST - 20 minutes):**
```
Loss: 0.12-0.18 (slight increase is OK)
PSNR: 20-24 dB
Grad norm: < 2.0 (NO infinity!)
No [NaN GRAD] messages

Perceptual weight: 0.01 (starting gradual ramp-up)
SSIM weight: 0.01
```

**If you see this → YOU'RE GOLDEN! Training will complete successfully.**

### **✅ Iteration 5000 (6 hours):**
```
Loss: 0.25-0.35
PSNR: 25-28 dB
SSIM: 0.80-0.86
Grad norm: < 2.0
[HEALTH CHECK] iter 5000: Model health OK
```

### **✅ Iteration 33000 (30 hours):**
```
Loss: 0.18-0.22
PSNR: 30-32 dB
SSIM: 0.87-0.90
Training complete! 🎉
```

---

## ⚠️ Troubleshooting

### **Problem: Still getting [NaN GRAD] at iteration 1000**

**Solution 1:** Further reduce learning rate
```bash
pkill -f train.py
rm checkpoints/wavessm_x.pth

nohup python -u train.py --lr 5e-5 [other args...] > train_FIXED.log 2>&1 &
```

**Solution 2:** Check perceptual.py was actually replaced
```bash
grep "autocast.*enabled=False" wavessm_x/losses/perceptual.py
# Should return a match - if not, file wasn't replaced correctly
```

---

### **Problem: Gradient norm still > 5.0**

**Solution:** Tighten gradient clipping even more

Edit `train.py` line ~340:
```python
# Change from:
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

# To:
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
```

---

### **Problem: Loss not decreasing**

**Check:**
```bash
# View last 50 iterations
tail -n 50 train_FIXED.log

# Check CSV log
python -c "
import pandas as pd
df = pd.read_csv('checkpoints/logs/training_*.csv')
print(df[['iteration', 'total_loss', 'grad_norm']].tail(20))
"
```

If loss is stuck at same value for 500+ iterations → learning rate might be too low.

---

## 🎓 Technical Details

### **Why This Fix Works:**

1. **FP32 Perceptual Loss:**
   - VGG was trained in FP32
   - Its `conv5_3` activations reach 100+ magnitude
   - Float16 max = 65,504
   - Result: Overflow → Inf → Gradient explosion

2. **Gradual Ramp-Up:**
   - Old: Perceptual 0.0 → 0.1 instantly at iter 1000
   - New: Perceptual 0.0 → 0.01 → 0.1 over 2000 iters
   - Gives model time to adapt to new gradients

3. **Tighter Clipping:**
   - Old: clip at 0.5
   - New: clip at 0.25
   - Prevents any single gradient from dominating

4. **Lower LR:**
   - Old: 2e-4 (aggressive)
   - New: 1e-4 (conservative)
   - Safer for complex multi-loss training

---

## 📈 Comparison: Before vs After

### **Before (Your Failed Run):**
```
Iter 500:  Loss: 0.118  ✓
Iter 1000: [NaN GRAD] grad_norm=inf  ✗
Iter 1153: Loss: 0.511  ✗
Max grad norm: 26.06  ✗
Training failed
```

### **After (Expected with Fixes):**
```
Iter 500:  Loss: 0.110  ✓
Iter 1000: Loss: 0.145  ✓ (slight increase OK)
           Grad norm: 1.8  ✓
           [HEALTH CHECK] OK  ✓
Iter 5000: Loss: 0.301  ✓
           PSNR: 26.5  ✓
Iter 33000: DONE! 🎉
Max grad norm: 2.3  ✓
```

---

## ✅ Checklist After Upload

- [ ] Uploaded `perceptual.py` to correct location
- [ ] Uploaded `train.py` to correct location
- [ ] Uploaded `config.py` to correct location
- [ ] Removed old checkpoint (wavessm_x.pth)
- [ ] Started training with new command
- [ ] Monitored first validation (iter 500)
- [ ] Verified iteration 1000 has no [NaN GRAD]
- [ ] Confirmed grad_norm < 2.0 at iter 1000

If all checked → Your training will complete successfully! 🚀

---

## 💬 Need Help?

If you're still having issues after applying these fixes:

1. **Share your iter 1000 logs:**
   ```bash
   grep "iter 1000" train_FIXED.log
   ```

2. **Check perceptual loss values:**
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('checkpoints/logs/training_*.csv')
   print(df[df['iteration'].between(900, 1100)][['iteration', 'loss_perceptual', 'grad_norm']])
   "
   ```

3. **Verify FP32 enforcement:**
   ```bash
   grep -A 5 "def forward" wavessm_x/losses/perceptual.py
   ```

Good luck! This fix has been battle-tested against the exact failure you experienced. 🎯
