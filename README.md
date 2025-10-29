
# Fraud Detection in Instant Payment Systems

This project demonstrates **real-time fraud detection** in Instant Payment Systems (IPS) using **machine learning**.  
It implements **supervised**, **unsupervised**, and **semi-supervised** models to compare accuracy, detection time, and overall performance.  
The application is built with **Streamlit** and uses simulated transaction data to replicate a live streaming environment.

🔗 **Live Demo:** [https://yourusername-frauddetector.streamlit.app](https://yourusername-frauddetector.streamlit.app)  
📂 **Source Code:** [https://github.com/sharhaynes/Fraud-Detection](https://github.com/sharhaynes/Fraud-Detection)

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

## Running the Application Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/sharhaynes/Fraud-Detection.git
   cd Fraud-Detection
*
