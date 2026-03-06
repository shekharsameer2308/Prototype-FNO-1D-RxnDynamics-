# FNO Demo — IP0SB0200004
## Neural Operator Surrogate for Parametric 1D Reaction–Diffusion Dynamics

**Project:** IP0SB0200004 | **Supervisor:** Shubhangi Bansude | **Intern:** Sameer Shekhar, BIT Mesra

---

## What This Is

A fully interactive browser demo of the FNO surrogate project. It includes:

- **Real Crank-Nicolson PDE solver** running in the browser (JavaScript)
- **FNO surrogate prediction** (analytical approximation of the trained model)
- **Training simulation** with animated loss curves and epoch log
- **Benchmarking** — speedup vs batch size across 7 batch sizes
- **Model evaluation** — 20 in-distribution + 5 OOD test cases
- **Full source code** for the real Python project (solver, FNO, training script)
- **AI explanation** of simulation results (requires Anthropic API key)

---

## Quick Start

### Step 1 — Install Node.js (if not already installed)

Download from: https://nodejs.org/en/download
Choose the **LTS** version. After install, verify:

```bash
node --version    # should be v18 or higher
npm --version     # should be v9 or higher
```

### Step 2 — Install dependencies

Open a terminal in this folder and run:

```bash
npm install
```

This will install React and all required packages. It may take 1–2 minutes.

### Step 3 — Start the development server

```bash
npm start
```

The app will open automatically at **http://localhost:3000**

---

## Using the App

### ⚗️ Simulate Tab
1. Adjust the 4 sliders on the left (D, r, μ, σ)
2. Click **▶ Run Simulation**
3. The Crank-Nicolson solver computes the ground-truth solution
4. The FNO surrogate predicts the final state instantly
5. Compare the charts, metrics, and error field
6. Click **✨ Explain** for an AI explanation (requires API key — see below)

### 🧠 Training Tab
- Click **🧠 Simulate Training** in the sidebar
- Watch the animated training loop with loss curves and epoch log
- View the full FNO architecture and JSON config

### 📊 Benchmark Tab
- Click **📊 Run Benchmark** in the sidebar
- See speedup vs batch size across 7 batch sizes
- Check whether the 50× SOP target is met

### 🔍 Evaluate Tab
- Click **🔍 Evaluate Model** in the sidebar
- View accuracy on 20 in-distribution test cases
- View OOD generalisation on 5 out-of-distribution cases
- Check the Gate 4 acceptance criteria

### 🗂️ Dataset Tab
- View parameter space specification
- View dataset generation code (Python)
- View HDF5 file structure

### </> Code Tab
- Full Python solver code (Crank-Nicolson)
- Full FNO model code (PyTorch + neuraloperator)
- Full training script with SEED=42
- Environment setup commands

---

## AI Explanation Feature

The "✨ Explain" button calls the Anthropic Claude API to explain the simulation results.

To enable it:

1. Get an API key from: https://console.anthropic.com
2. Open `src/App.jsx`
3. Find the `getAI` function (search for `getAI`)
4. The API call is already set up — it uses the key from your environment or directly via the fetch call

> **Note:** In the claude.ai artifact environment the API call works automatically.
> In VS Code you need a valid Anthropic API key. The rest of the app works without it.

---

## Project Structure

```
fno_project/
├── public/
│   └── index.html          # HTML entry point
├── src/
│   ├── index.js            # React entry point
│   └── App.jsx             # Full application (solver + FNO + UI)
├── package.json            # Dependencies
└── README.md               # This file
```

---

## The Real Python Project

The **</> Code** tab shows the actual Python code you would run for the real project:

### Environment Setup
```bash
# Create conda environment
conda create -n fno_rd python=3.11
conda activate fno_rd

# Install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install neuraloperator h5py numpy scipy matplotlib jupyter tensorboard
```

### Run Solver Tests
```bash
pytest tests/test_solver.py -v
```

### Generate Dataset
```bash
python data/generate_dataset.py
```

### Train FNO
```bash
python model/train.py --config configs/run_20260315_001.json
```

### Monitor Training
```bash
tensorboard --logdir runs/
```

### Evaluate
```bash
python scripts/evaluate.py --checkpoint model/best_model.pt
```

---

## The PDE

The Fisher-KPP reaction–diffusion equation:

```
∂u/∂t = D · ∂²u/∂x²  +  r · u · (1 - u)
```

- **D** — diffusion coefficient (how fast species spreads)
- **r** — reaction rate (how fast the reaction proceeds)
- **u** — species concentration ∈ [0, 1]
- Boundary conditions: zero-flux Neumann at both ends
- Initial condition: Gaussian centred at μ with width σ

Analytical wave speed: **c = 2√(D·r)**

---

## FNO Architecture

```
Input: (u₀(x), D, r) → shape (B, 3, 128)
  ↓
Lifting layer: Linear(3 → 64)
  ↓
FNO Layer 1: SpectralConv1d(64, 64, 32) + W(64, 64)  + GELU
FNO Layer 2: SpectralConv1d(64, 64, 32) + W(64, 64)  + GELU
FNO Layer 3: SpectralConv1d(64, 64, 32) + W(64, 64)  + GELU
FNO Layer 4: SpectralConv1d(64, 64, 32) + W(64, 64)  + GELU
  ↓
Projection: Linear(64 → 128) → GELU → Linear(128 → 1)
  ↓
Output: u(x, T=1.0) → shape (B, 1, 128)

Total parameters: ~580,000
```

---

## Acceptance Criteria (from SOP-2026-004)

| Criterion | Target |
|-----------|--------|
| Mean relative L2 error (test set) | < 1.0% |
| Max relative L2 error (test set) | < 5.0% |
| Speedup vs direct solver (batch=100) | ≥ 50× |
| OOD generalisation (1.5× parameter range) | < 10% |

---

## References

- Li, Z. et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021.
- neuraloperator library: https://github.com/neuraloperator/neuraloperator
- Fisher, R.A. (1937). *The wave of advance of advantageous genes*. Annals of Eugenics.
