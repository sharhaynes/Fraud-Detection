
# Fraud Detection in Instant Payment Systems

This project demonstrates **real-time fraud detection** in Instant Payment Systems (IPS) using **machine learning**.  
It implements **supervised**, **unsupervised**, and **semi-supervised** models to compare accuracy, detection time, and overall performance.  
The application is built with **Streamlit** and uses simulated transaction data to replicate a live streaming environment.

---

## Overview

The system evaluates multiple machine learning approaches for identifying fraudulent transactions in IPS data.  
Users can upload datasets, train models, visualize performance metrics, and measure detection time across models.

---

## Features

- Evaluation of supervised, unsupervised, and semi-supervised models  
- Comparison of model accuracy, precision, recall, F1-score, and AUC  
- Measurement of detection time per transaction (based on simulation steps)  
- Visualization of confusion matrices and ROC curves  
- Streamlit-based interactive interface for model testing and analysis  
- Synthetic data generation for simulation and benchmarking  

---

## Tech Stack

| Category | Tools |
|-----------|--------|
| Language | Python |
| Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Models Implemented

- **Supervised Learning:** Logistic Regression, Decision Tree, Random Forest, Support Vector Machine  
- **Unsupervised Learning:** Isolation Forest, One-Class SVM  
- **Semi-Supervised Learning:** Label Propagation / Self-training Classifier  

---
## Live Demos

| Type | Live Demo | Source Code |
|------|------------|--------------|
| 🧩 Supervised | [Streamlit App](https://your-supervised.streamlit.app) | [supervised_app](./supervised_app) |
| 🌀 Unsupervised | [Streamlit App](https://your-unsupervised.streamlit.app) | [unsupervised_app](./unsupervised_app) |
| 🔄 Semi-Supervised | [Streamlit App](https://your-semisupervised.streamlit.app) | [semi_supervised_app](./semi_supervised_app) |

---

## License
This project is licensed under the **Apache 2.0 License** — see the [LICENSE](./LICENSE) file for details.

---

## Author
**T'Shara Haynes**  
📧 [haynestshara0@gmail.com]
🌐 [yourportfolio.github.io](https://yourportfolio.github.io)

## Running the Application Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/sharhaynes/Fraud-Detection.git
   cd Fraud-Detection
*
