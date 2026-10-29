# 🚀 Manasitra Custom Model — Google Colab GPU Training Guide

Google Colab par **NVIDIA T4 / V100 / A100 GPU** ka use karke aap Manasitra ke custom psychological models ko **10x se 20x fast** train kar sakte hain.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashpatadiya1724-cmyk/mansitra-ai-or/blob/main/manasitra/ai-engine/custom_model/manasitra_train_colab.ipynb)

---

## ⚡ Google Colab Par Train Karne Ke Fayde (Benefits)

1. **Free Cloud GPU (NVIDIA T4 - 16GB VRAM):**
   - Laptop ka CPU 100% engage nahi hoga.
   - Laptop garam nahi hoga aur battery waste nahi hogi.
2. **CUDA FP16 Automatic Mixed Precision (AMP):**
   - Hardware Tensor Cores use hote hain, jisse per-epoch time **15 se 25 seconds** ho jata hai.
3. **Multi-Scale Architecture Options:**
   - **`small` (32M Parameters):** ~5 minute me train ho jata hai.
   - **`base` (146.5M Parameters):** ~15-20 minute me train ho jata hai.
   - **`large` (371.6M Parameters):** ~40-45 minute me train ho jata hai.
   - **`xlarge` (758.3M Parameters):** ~1.5 ghante me heavy foundation model train ho jata hai.
4. **Direct Google Drive Integration:**
   - Trained model weights (`manasitra_model.pt`, `model_metadata.json`, `tokenizer.json`) direct aapke Google Drive me backup ho jaate hain.

---

## 📋 Step-by-Step Execution Guide

### Step 1: Open Notebook in Google Colab
1. Browser me [Google Colab](https://colab.research.google.com/) open karein.
2. **GitHub** tab choose karein aur paste karein:
   ```text
   https://github.com/yashpatadiya1724-cmyk/mansitra-ai-or
   ```
3. Notebook select karein:
   ```text
   manasitra/ai-engine/custom_model/manasitra_train_colab.ipynb
   ```

### Step 2: Enable GPU Hardware
1. Top menu me **Runtime** par click karein.
2. **Change runtime type** select karein.
3. **Hardware accelerator** me **T4 GPU** select karke **Save** karein.

### Step 3: Run Setup & Verification Cells
- **Cell 1 (`!nvidia-smi`):** Verify karein ki T4 GPU active hai (15-16 GB VRAM dikhega).
- **Cell 2 (`git clone`):** Repository automatically clone ho jayegi.

### Step 4: Choose Model Size (Dropdown Form)
Notebook me interactive dropdown form milega:
- `MODEL_PROFILE`: Select karein **`large`** (372M params) ya **`base`** (146M params).
- `EPOCHS`: **100** (recommended).
- `BATCH_SIZE`: **32** (T4 GPU par easily fit hota hai).
- `SAVE_TO_DRIVE`: Agar Google Drive me direct save karna hai to checkbox tick karein.

### Step 5: Start Training
- Cell 4 run karein (`train.train()`).
- Live training progress, Cosine Annealing learning rate, loss decay aur perplexity screen par print hoga.
- Best model automatically `saved_model/manasitra_model.pt` me save hoga.

### Step 6: Test & Download
- **Live Test Cell:** Notebook ke andar hi 5 empathetic test questions run karke response verify karein.
- **Export Cell:** Ek click se `manasitra_large_model.zip` aapke laptop me download ho jayegi.

---

## 💻 Downloaded Model Ko Apne Laptop Me Kaise Use Karein?

Download hui zip file ko extract karke files ko yahan paste karein:
```text
manasitra/
└── ai-engine/
    └── custom_model/
        └── saved_model/
            ├── manasitra_model.pt
            ├── model_metadata.json
            └── tokenizer.json
```

Aapka local backend restart hote hi naye trained weights ko zero-copy memory mapping ke sath **sub-second** speed me serve karne lagega!
