# Time Series Prediction for Digital Waveform in Digital Circuits
**Developed for: Cadence**

## 👥 Team Members
| Name | University | GitHub Handle |
| :--- | :--- | :--- |
| Erica Kim | Tufts University | [@EricaMKim] |
| [Name 2] | | [@handle] | 
| [Name 3] | | [@handle] | 
| [Name 4] | | [@handle] |

## 🌟 Project Highlights
* **Achieved 97% Accuracy:** The final Long Short-Term Memory (LSTM) model achieved the best performance with an accuracy of 97% and an RMSE of 0.076 for waveform prediction.
* **Comprehensive RNN Comparison:** We developed and compared five different RNN architectures (Vanilla, GRU, LSTM, Seq2Seq, and Transformer) to identify the optimal solution for this time-series task.
*  **Custom Data Preprocessing:** A custom pipeline was developed to reformat and extract square-shaped voltage signals, removing redundant regions to create ML-ready time-series data

## 📖 Project Overview
**Objective & Scope:**
The main objective of this AI Studio Final Project was to predict digital waveform time-series in digital circuits using Recurrent Neural Network (RNN) models. This aims to improve timing analysis and signal integrity within circuit design. Our scope involved preprocessing raw I/O voltage waveform data, developing and comparing various RNN architectures, and evaluating their performance.

**Business Relevance:**
This prediction model is highly relevant to Cadence's operations as it helps to **reduce simulation time and cost** associated with traditional methods. Furthermore, it **improves reliability** by catching potential circuit errors early and can be used to **discover hidden circuit behaviors** that standard simulations might miss.

## 📂 Repository Structure (NOT DONE)
The repository is organized into the following directories:
```text

├── models/
│   └── transformer/            # Transformer model updates and files
├── scripts/                    # Utility scripts (e.g., LSTM model generation)
├── Seq2Seq_Model.py            # Source code for the Seq2Seq architecture
├── gru_model.py                # Source code for the GRU architecture
├── waveforms.csv               # Raw voltage waveform dataset
├── metadata.csv                # Circuit metadata (sim_ids, parameters)
├── processed_waveforms.csv     # Preprocessed data with feature engineering
├── downsampled_waveforms.csv   # Data with redundant flat regions removed
├── full_waveforms.zip          # Compressed archive of the full dataset
├── correlation_heatmap.png     # Visualization of feature correlations (EDA)
└── README.md                   # Project documentation
```

## 📊 Data Exploration (EDA)
**Dataset Description:**
* **Source:** The project used the "Waveforms Dataset" and referenced external works on signal prediction in digital circuits (e.g., Salzmann, DATE 2025).
* **Structure:** The data is **time-series** voltage waveforms, combined with physical circuit parameters (PVT, load, input). The core data files were `Waveforms.csv` and `metadata.csv`.

**Preprocessing & Cleaning:**
* We **merged** `Waveforms.csv` and `metadata.csv` using the `sim_id` key.
* Categorical features like `gate` and `process` were **one-hot encoded**.
* Numerical values were **normalized** using standard scaling.
* A custom preprocessing pipeline was developed to extract **square-shaped voltage signals** (0–1.2 V), using downsampling and masking to remove redundant flat regions and focus on valid signal transition points.

---

## 🧠 Model Development
**Approach & Justification:**
We selected **Recurrent Neural Networks (RNNs)** because they are inherently suited for **time-series prediction** and are designed to capture temporal dependencies in sequential voltage data. We used **PyTorch/TensorFlow** for implementation.

**Architectures Evaluated:**
* **LSTM (Long Short-Term Memory):** Chosen for its robustness and ability to capture **long-term temporal dependencies**; it ultimately performed best.
* **GRU (Gated Recurrent Unit):** Evaluated for its balance of performance and computational efficiency.
* **Vanilla RNN:** Served as a baseline, but struggled with longer sequences due to gradient issues.
* **Seq2Seq:** Tested due to it's variable-length sequence handling, but found to perform poorly with square waveforms. Additionally, was extremely slow when runnning. 
* **Transformer:** Evaluated for its self-attention mechanism but found to be computationally expensive and prone to overfitting.

---

## 📈 Results & Key Findings
**Model Comparison (Performance Metrics):**

| Model | Accuracy | MSE | **RMSE** | MAE |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM** | **97%** | **0.0058** | **0.076** | **0.035** |
| GRU | 68.75% | 0.0070 | 0.0836 | 0.0561 |
| Transformer | 61.83% | 0.13 | 0.17 | 0.21 |
| Vanilla RNN (Baseline) | ~55–60% | 0.0627 | 0.25 | 0.22 |
| Seq2Seq | 43.97% | 0.299 | 0.547 | 0.347 |

**Key Findings:**
* The **LSTM model achieved the best performance** with 97% accuracy and the lowest RMSE of 0.076.
* The **majority of the models struggled** to accurately reproduce the sharp edges of the square waveform shape.
* Some model predictions resulted in a **flat line at 0 or 1 volt** (failure modes) instead of the true dynamic signal.

---

## 💭 Discussion & Reflection (What Didn't Work)
**Challenges & Limitations:**
* **Synthetic Data:** The model was trained on **ChatGPT-generated data** which contained flaws like instantaneous changes in output voltage, resulting in unrealistic waveforms. This limited the overall performance ceiling.
* **Architecture Flaws:**
    * **Transformers** suffered from **overfitting** and were computationally expensive.
    * **Seq2Seq** performance dropped on **long sequences** and had slow decoding.
    * **Vanilla RNN** struggled with **gradient issues** on longer sequences.

---

## 🚀 Next Steps
1.  **Improve Data Quality:** Apply our successful models (LSTM, GRU) to **better, non-synthetic data** to significantly improve prediction accuracy and generalization.
2.  **Fine-Tuning:** Conduct extensive **hyperparameter tuning** on the best-performing models to maximize results on the new, higher-quality dataset.
3.  **Industrial Automation:** Develop a system to use the prediction model to **automatically decide if output voltages** in electronic circuits are fit for industrial standards.
