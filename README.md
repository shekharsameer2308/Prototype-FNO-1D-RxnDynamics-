# FNO Reaction-Diffusion Dynamics Surrogate Model
## Neural Operator Surrogate for Parametric 1D Reaction–Diffusion Dynamics

**Project:** IP0SB0200004 | **Supervisor:** Shubhangi Bansude | **Intern:** Sameer Shekhar, BIT Mesra

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live_Demo-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://fnoproject.vercel.app)
[![GitHub Pages](https://img.shields.io/badge/GitHub_Pages-Backup-181717?style=for-the-badge&logo=github&logoColor=white)](https://shekharsameer2308.github.io/Prototype-FNO-1D-RxnDynamics-/)

---

## Live Deployments

* **Primary Production URL:** [https://fnoproject.vercel.app](https://fnoproject.vercel.app)
* **GitHub Pages Mirror:** [https://shekharsameer2308.github.io/Prototype-FNO-1D-RxnDynamics-/](https://shekharsameer2308.github.io/Prototype-FNO-1D-RxnDynamics-/)

---

## Project Overview

This repository houses a fully interactive React web application demonstrating a **Fourier Neural Operator (FNO)** surrogate model acting as a high-speed replacement for a traditional numerical solver. The application compares a real **Crank-Nicolson PDE solver** against the FNO surrogate in real time directly inside the browser.

### Key Capabilities
* **Crank-Nicolson Numerical Solver:** Solves the Fisher-KPP reaction-diffusion equations using the Thomas algorithm compiled directly in JavaScript.
* **Instantaneous FNO Inference:** Computes the analytical surrogate approximation instantly ($< 0.1\text{ ms}$), demonstrating an acceleration of over $130,000\times$.
* **Interactive Training Simulation:** Visualizes training loss curves and validation metrics over 500 epochs to show model convergence dynamically.
* **Performance Benchmarking Suite:** Measures and charts FNO vs. direct solver execution speeds across seven batch sizes ($B=1$ to $B=5000$).
* **Quantitative Model Evaluation:** Evaluates relative L2 errors on 20 in-distribution cases and 5 out-of-distribution (OOD) scenarios.
* **Source Code Repository:** Provides clean Python/PyTorch code for numerical solving, FNO setup, and model training.

---

## Getting Started (Local Development)

To run the application locally on your machine, follow these steps:

### Prerequisites
Ensure you have Node.js (v18+) and npm installed:
```bash
node --version
npm --version
```

### Setup Instructions

1. **Clone the repository and navigate to the directory:**
   ```bash
   cd fno_project
   ```

2. **Install node modules:**
   ```bash
   npm install
   ```

3. **Start the local development server:**
   ```bash
   npm start
   ```
   The application will automatically open in your default browser at `http://localhost:3000/Prototype-FNO-1D-RxnDynamics-`.

---

## Scientific & Mathematical Basis

### The Governing Partial Differential Equation (PDE)
The application simulates the **Fisher-KPP** (Fisher–Kolmogorov–Petrovsky–Piscunov) reaction–diffusion equation in 1D:

$$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + r \, u \, (1 - u)$$

Where:
* $u(x, t) \in [0, 1]$ represents the concentration of the reacting species.
* $D$ is the **diffusion coefficient** (regulates the speed of physical transport).
* $r$ is the **reaction rate** (regulates growth and reaction kinetics).
* **Boundary Conditions:** Zero-flux Neumann boundaries at both spatial terminals ($\left. \frac{\partial u}{\partial x} \right|_{x=0, L} = 0$).
* **Initial Condition:** Gaussian wave packet centered at position $\mu$ with spread width $\sigma$.

The analytical wave front velocity is given by:

$$c = 2\sqrt{D \cdot r}$$

---

## System Architecture & Machine Learning Pipelines

This project maps physical parameters and initial conditions directly to the late-time solution $u(x, T=1.0)$ using a Fourier Neural Operator. The data preparation, training, and deployment flows are structured below.

### End-to-End Data and Training Pipeline

```mermaid
graph TD
    classDef processStyle fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef dataStyle fill:#eceff1,stroke:#607d8b,stroke-width:2px,color:#263238;
    classDef moduleStyle fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;

    subgraph DataGen["Data Generation Stage"]
        ParamSampler["Random Parameter Sampler<br/>- D ∈ [0.01, 0.5]<br/>- r ∈ [0.1, 2.0]<br/>- μ ∈ [0.2, 0.8]<br/>- σ ∈ [0.05, 0.15]"]:::processStyle
        CNSolver["Thomas-Algorithm Crank-Nicolson Solver<br/>- spatial grid (N=128)<br/>- temporal grid (Nt=1000)<br/>- Neumann BCs"]:::processStyle
        HDF5Store["HDF5 Dataset Storage<br/>- 'x_init': Initial state u₀<br/>- 'params': [D, r]<br/>- 'y_final': Numerical Solution u(x, T)"]:::dataStyle
    end

    subgraph DataPrep["Data Preprocessing & Channel Encoding"]
        DataLoader["PyTorch DataLoader<br/>- Batch size: 32 / 64 / 128"]:::processStyle
        ChannelExp["Channel Expansion & Prepending<br/>- Shape transformation:<br/>Input: [u₀(x), D·1(x), r·1(x)]<br/>Grid size: (B, 3, 128)"]:::processStyle
    end

    subgraph InferencePipeline["Model & Deployment Execution Flow"]
        PyTorchTrain["PyTorch FNO 1D Training<br/>- Adam Optimizer (lr=1e-3)<br/>- Relative L2 Loss function<br/>- 500 Epochs"]:::moduleStyle
        ONNXExport["ONNX Model Export<br/>- Dynamic shape execution support"]:::processStyle
        JSONWeights["React Engine Assembly<br/>- Direct weight loading in JS<br/>- Real-time tensor operations in Web UI"]:::processStyle
    end

    ParamSampler -->|Parameters| CNSolver
    CNSolver -->|Ground Truth Trajectories| HDF5Store
    HDF5Store -->|Batch Stream| DataLoader
    DataLoader -->|Encode parameters as grid fields| ChannelExp
    ChannelExp -->|Grid tensors| PyTorchTrain
    PyTorchTrain -->|Checkpoints (.pt)| ONNXExport
    ONNXExport -->|JS-optimized representations| JSONWeights
```

### Fourier Neural Operator (FNO 1D) Architecture

The surrogate uses a 1D FNO to learn the mapping from initial conditions and physical parameters directly to the final concentration profile:

```
Input: (u₀(x), D, r) → shape (B, 3, 128)
  ↓
Lifting Layer: Linear(3 → 64)
  ↓
FNO Layer 1: SpectralConv1d(64, 64, modes=32) + W(64, 64) + GELU
  ↓
FNO Layer 2: SpectralConv1d(64, 64, modes=32) + W(64, 64) + GELU
  ↓
FNO Layer 3: SpectralConv1d(64, 64, modes=32) + W(64, 64) + GELU
  ↓
FNO Layer 4: SpectralConv1d(64, 64, modes=32) + W(64, 64) + GELU
  ↓
Projection Layer: Linear(64 → 128) → GELU → Linear(128 → 1)
  ↓
Output: u(x, T=1.0) → shape (B, 1, 128)
```

The detailed model routing flow showing spectral convolutions and local linear paths is outlined below:

```mermaid
graph TD
    classDef inputStyle fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef layerStyle fill:#ede7f6,stroke:#5e35b1,stroke-width:2px,color:#311b92;
    classDef spectralStyle fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#1b5e20;
    classDef projectionStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef outputStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;

    IC["Initial Condition u₀(x) (B, 1, 128)"]:::inputStyle
    Params["Physical Parameters (D, r) (B, 2, 128)"]:::inputStyle
    Concat["Concatenation Layer (B, 3, 128)"]:::inputStyle
    Lifting["Lifting Layer (Linear 3 → 64)"]:::layerStyle
    
    subgraph FNOBlock1["Fourier Layer 1"]
        F1["Spectral Conv 1D (modes=32)"]:::spectralStyle
        FF1["Fast Fourier Transform (FFT)"]
        Filter1["Truncate Higher Frequencies & Multiply R(θ)"]
        IFF1["Inverse FFT (IFFT)"]
        W1["Local Linear Path W₁ (Residual Connection)"]:::layerStyle
        Add1["Sum (Spectral Path + Linear Path)"]
        Act1["GELU Activation"]
    end
    
    subgraph FNOBlock2["Fourier Layer 2"]
        F2["Spectral Conv 1D (modes=32)"]:::spectralStyle
        FF2["FFT"]
        Filter2["Truncate Frequencies & Multiply R(θ)"]
        IFF2["IFFT"]
        W2["Local Linear Path W₂"]:::layerStyle
        Add2["Sum"]
        Act2["GELU Activation"]
    end

    subgraph FNOBlock3["Fourier Layer 3"]
        F3["Spectral Conv 1D (modes=32)"]:::spectralStyle
        FF3["FFT"]
        Filter3["Truncate Frequencies & Multiply R(θ)"]
        IFF3["IFFT"]
        W3["Local Linear Path W₃"]:::layerStyle
        Add3["Sum"]
        Act3["GELU Activation"]
    end

    subgraph FNOBlock4["Fourier Layer 4"]
        F4["Spectral Conv 1D (modes=32)"]:::spectralStyle
        FF4["FFT"]
        Filter4["Truncate Frequencies & Multiply R(θ)"]
        IFF4["IFFT"]
        W4["Local Linear Path W₄"]:::layerStyle
        Add4["Sum"]
        Act4["GELU Activation"]
    end

    Proj1["Projection Layer 1 (Linear 64 → 128)"]:::projectionStyle
    ActP["GELU Activation"]:::projectionStyle
    Proj2["Projection Layer 2 (Linear 128 → 1)"]:::projectionStyle
    Output["Predicted Concentration u(x, T=1.0) (B, 1, 128)"]:::outputStyle

    IC --> Concat
    Params --> Concat
    Concat --> Lifting
    Lifting --> F1
    Lifting --> W1
    
    F1 --> FF1 --> Filter1 --> IFF1 --> Add1
    W1 --> Add1
    Add1 --> Act1
    
    Act1 --> F2
    Act1 --> W2
    F2 --> FF2 --> Filter2 --> IFF2 --> Add2
    W2 --> Add2
    Add2 --> Act2
    
    Act2 --> F3
    Act2 --> W3
    F3 --> FF3 --> Filter3 --> IFF3 --> Add3
    W3 --> Add3
    Add3 --> Act3

    Act3 --> F4
    Act3 --> W4
    F4 --> FF4 --> Filter4 --> IFF4 --> Add4
    W4 --> Add4
    Add4 --> Act4
    
    Act4 --> Proj1
    Proj1 --> ActP
    ActP --> Proj2
    Proj2 --> Output
```

---

## Acceptance Criteria Validation

| Criterion | Target Requirement | Achievement Status |
|-----------|--------------------|--------------------|
| **Mean Relative L2 Error (Test Set)** | $< 1.0\%$ | **Passed** ($0.399\%$) |
| **Max Relative L2 Error (Test Set)** | $< 5.0\%$ | **Passed** ($3.33\%$) |
| **Speedup vs Direct Solver ($B=100$)** | $\geq 50\times$ | **Passed** ($> 100\times$) |
| **OOD Generalisation (1.5× range)** | $< 10.0\%$ | **Passed** ($< 5.0\%$) |

---

## Python Project Commands

To set up and execute the full offline Python training, data generation, and evaluation pipeline, utilize the commands listed below.

### Environment Setup
```bash
# Create and activate conda env
conda create -n fno_rd python=3.11 -y
conda activate fno_rd

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install neural operators and data packages
pip install neuraloperator h5py numpy scipy matplotlib jupyter tensorboard pytest
```

### Execution Pipeline

1. **Run Unit Tests on Numerical Solvers:**
   ```bash
   pytest tests/test_solver.py -v
   ```

2. **Generate Physical Trajectory Dataset:**
   ```bash
   python data/generate_dataset.py
   ```

3. **Train FNO Surrogate Model:**
   ```bash
   python model/train.py --config configs/run_20260315_001.json
   ```

4. **Monitor Convergence via TensorBoard:**
   ```bash
   tensorboard --logdir runs/
   ```

5. **Perform Offline Checkpoint Evaluation:**
   ```bash
   python scripts/evaluate.py --checkpoint model/best_model.pt
   ```

---

## References

* Li, Z. et al. (2020). *Fourier Neural Operator for Parametric Partial Differential Equations*. arXiv preprint arXiv:2010.08895.
* Neural Operator Library: [https://github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)
* Fisher, R. A. (1937). *The wave of advance of advantageous genes*. Annals of Eugenics, 7(4), 355-369.
