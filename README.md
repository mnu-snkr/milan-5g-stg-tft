# Milan 5G Traffic Forecasting: Spatio-Temporal Graph TFT

## 1. Architectural Overview
To forecast dynamic telecommunications traffic across the Milan grid, this architecture injects a custom Temporal Graph Convolution (GCN) layer into a baseline Temporal Fusion Transformer (TFT). This hybrid approach was engineered to mathematically capture hidden spatial network dependencies—how a surge in one neighborhood physically impacts adjacent cell towers—rather than relying solely on isolated temporal patterns.

## 2. Cloud CI/CD & Data Gravity
This repository maintains a sterile CI/CD pipeline by strictly isolating the source code from heavy data artifacts. While the Python architecture is version-controlled via Git, the multi-gigabyte `.parquet` datasets and compiled PyTorch `.ckpt` binary weights are hosted on Kaggle and dynamically injected at runtime. This architecture bypasses Git's strict binary limits and completely prevents repository bloat.

## 3. Evaluation Metrics & Known Bottlenecks
*   **University Grid (4259):** MAE 19.66
*   **City Center Hub (5060):** MAE 185.21
*   **Residential Grid (4456):** MAE 75.33

**Architectural Bottleneck (GCN Oversmoothing):**
During evaluation, the model exhibited a severe "Phase Shift" and "Parasitic Gain" phenomenon. By forcing distinct grid nodes to communicate via a Pearson correlation adjacency matrix, the network suffered from GCN oversmoothing. The graph layers diluted unique local signals, causing the model to incorrectly predict massive traffic surges in the University grid during known dead periods simply because it was reacting to overlapping traffic from distant, unrelated City Center nodes.

## 4. Execution Pipeline
To replicate this environment locally or on a cloud GPU:

```bash
git clone [https://github.com/mnu-snkr/milan-5g-stg-tft.git](https://github.com/mnu-snkr/milan-5g-stg-tft.git)
cd milan-5g-stg-tft
python src/evaluate.py