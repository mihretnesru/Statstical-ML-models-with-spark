# 🧠 Student Performance Prediction with PySpark

This project predicts student exam performance using **Apache Spark** for building efficient and scalable machine learning models. By leveraging distributed computing, the goal is to enhance processing efficiency while analyzing the factors influencing student success, such as study habits, attendance, and parental involvement.

---

## 📜 Overview

Educational institutions often face challenges in identifying factors affecting student performance. This project aims to:
1. Predict whether a student will **pass or fail** an exam based on various features.
2. Demonstrate the efficiency of distributed computing with **Apache Spark**.
3. Compare performance using:
   - **Single-worker setup** (one VM as a worker).
   - **Distributed setup** (two VMs: one master and one worker).

Using the **Student Performance Factors Dataset** from Kaggle, this study highlights the computational advantages of Spark and provides insights into critical factors affecting student outcomes.

---

## ✨ Key Features

1. **Apache Spark for Distributed Computing**:
   - Optimized for scalability and speed.
   - Comparisons between single-worker and distributed-worker setups to evaluate computational efficiency.

2. **Comprehensive Data Pipeline**:
   - Data cleaning: Imputation of missing values (mean for numeric, mode for categorical).
   - Categorical encoding using **OneHotEncoder**.
   - Feature scaling and vectorization with **StandardScaler**.

3. **Statistical Machine Learning Models**:
   - Models: **Linear Support Vector Classifier (SVM)** and **Random Forest Classifier**.
   - Hyperparameter tuning with **CrossValidator** and **ParamGridBuilder**.

4. **Dataset Insights**:
   - Analysis of 20 features across 6,607 records, including:
     - Study habits
     - Attendance
     - Parental involvement
   - Creation of binary classification labels (`Pass` or `Fail`) based on exam scores.

5. **Performance Metrics**:
   - Accuracy, Precision, Recall, Specificity, and AUC score.
   - Visualization of ROC curves for model comparison.

---

## 🛠 Technology Stack

- **Apache Spark**: Distributed data processing and machine learning.
- **Python 3.x**: Programming language for building pipelines.
- **Matplotlib**: Visualization of ROC curves.
- **Scikit-learn**: Additional metrics for evaluation.

---


## 🔧 How to Run

### Prerequisites
- Two Virtual Machines (VMs) with Apache Spark installed:
  - **VM 1**: Master node.
  - **VM 2**: Worker node.
- Python 3.x installed on both VMs.
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

### Dataset
- **Source**: Kaggle - [Student Performance Factors Dataset]
- **Description**: 6,607 records with 20 features, focusing on elements affecting student performance in exams.


