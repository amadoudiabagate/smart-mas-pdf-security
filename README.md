# Hybrid Multi-Agent System for Malicious PDF Detection

This repository contains the source code and simulation framework for the article:

**"Hybrid Multi-Agent System for Automatic Detection of PDF Security Threats: Generative Data Augmentation and Supervised Machine Learning Approach"**

The system integrates a **modular multi-agent architecture** with **CTGAN-based data augmentation** and **supervised machine learning** to improve the detection of malicious PDF files.

---

## Project Structure

```
ProjectSimulation/
├── agents/
│   ├── uia.py       # UIA - User Interface Agent
│   ├── pla.py       # PLA - Planning Agent
│   ├── gaa.py       # GAA - Generative Augmentation Agent
│   ├── mla.py       # MLA - Machine Learning Agent
│   ├── upma.py      # UPMA - User Profile Management Agent
│   ├── dma.py       # DMA - Data Management Agent
│   └── __init__.py
├── analysis/
│   └── results_analysis.py   # Script to analyze simulation results and generate reports
├── model.py        # MAS orchestration and simulation loop
├── server.py       # Entry point to run a complete simulation
├── data/
│   └── synthetic_pdf_dataset.csv   # Synthetic dataset for testing
├── output/         # Simulation results (CSV, logs, figures)
├── requirements.txt
└── README.md
```

---

## Agents Overview

| Acronym | Agent Name | Function |
|---------|------------|----------|
| **UIA** | User Interface Agent | Entry point; manages user interactions, file intake, and output delivery |
| **PLA** | Planning Agent | Coordinates internal processes, assigns priorities, and execution order |
| **GAA** | Generative Augmentation Agent | Detects data imbalance and triggers CTGAN to generate synthetic samples |
| **MLA** | Machine Learning Agent | Performs supervised classification (Random Forest, etc.) and retraining |
| **UPMA** | User Profile Management Agent | Maintains risk profiles and user histories for adaptive behavior |
| **DMA** | Data Management Agent | Archives results, updates repositories, and prepares data for retraining/audit |

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mas-pdf-security.git
   cd mas-pdf-security/ProjectSimulation
   ```

2. Install dependencies (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

   Minimal requirements:
   - `mesa`
   - `scikit-learn`
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `xgboost` (if used)

---

## Running a Simulation

Run the main server script:
```bash
python server.py
```

Output:
- A CSV file will be generated in **`output/simulation_results_full.csv`**.
- This file contains the step-wise log of agent actions, predictions, and coordination.

You can then analyze the results:
```bash
python analysis/results_analysis.py
```

---

## Data and Code Availability

- **Code**: The full Python implementation of the MAS, CTGAN-based augmentation, and ML classifiers is available in this repository.  
- **Synthetic dataset**: An illustrative synthetic dataset is included to reproduce the pipeline structure (`synthetic_pdf_dataset.csv`).  
- **Full dataset**: Due to confidentiality and security restrictions, the complete dataset used in the study is **not publicly available**.  
- Aggregated or anonymized versions may be made available upon reasonable request to the corresponding author.

---

## Citation

If you use this code, please cite:

```
Diabagate A., et al.
Hybrid Multi-Agent System for Automatic Detection of PDF Security Threats: 
Generative Data Augmentation and Supervised Machine Learning Approach.
Submitted to Informatica, 2025.
```

---

## Author

- **Amadou Diabagate**  
University Felix Houphouët-Boigny  
Email: amadou1.diabagate@ufhb.edu.ci
