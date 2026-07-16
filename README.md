# vortextech-aiml-week2
# Student Mental Health Binary Classification

This project is a binary classification pipeline built using `scikit-learn` and `pandas` to predict whether a student experiences depression based on demographic, academic, and clinical indicators. 

The project satisfies the Week 2 AI & ML track requirements of the Vortex Tech Internship Program.

## 📌 Project Objective
The objective of this task is to transition from basic data analysis into supervised machine learning. By training a binary classification model (Logistic Regression), we predict a binary target outcome (`Do you have Depression?`) and evaluate the model's performance using standard classification metrics.

---

## 📊 Dataset Overview
The model uses the **Student Mental Health** dataset (`Student_Mental_health.csv`), which contains 101 records and 11 features detailing:
* **Demographics**: Gender, Age, Marital Status.
* **Academic Attributes**: Current Year of Study, CGPA ranges.
* **Mental Health Indicators**: Anxiety, Panic Attacks, and Specialist Treatment History.

### Preprocessing & Feature Engineering
Before training, the data was preprocessed to ensure optimal performance:
1.  **Irrelevant Column Drop**: Removed `Timestamp` and high-cardinality `What is your course?` to prevent severe overfitting.
2.  **String Standardization**: Unified values in `Your current year of Study` and `What is your CGPA?` (e.g., removing leading/trailing spaces and lowercase normalization).
3.  **Missing Values**: Imputed missing `Age` values using the dataset's median age.
4.  **Target Mapping**: Formatted the target column `Do you have Depression?` into binary numerical classes ($1$ for Yes, $0$ for No).
5.  **One-Hot Encoding**: Applied `pd.get_dummies(drop_first=True)` to convert categorical attributes into numerical dummy features.

---

## ⚙️ Model & Splitting
* **Train-Test Split**: The dataset was partitioned using an **80/20 train-test split** (80 samples for training, 21 samples for testing) with a fixed `random_state=42` to ensure reproducible results.
* **Algorithm**: **Logistic Regression** (configured with `max_iter=1000` to guarantee optimization convergence).

---

## 📈 Performance Results

Below is the evaluation of the trained Logistic Regression classifier on the unseen 20% test partition:

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | 0.7143 | 71.43% |
| **Precision** | 0.7500 | 75.00% |
| **Recall** | 0.3750 | 37.50% |
| **F1-Score** | 0.5000 | 50.00% |

### Key Takeaways & Limitations
* **The Recall Trade-off**: While the model achieves a strong precision score ($75\%$), its recall is quite low ($37.5\%$). This means that although the model is reliable when it *does* flag a student with depression, it misses a significant portion of actual positive cases. In clinical and medical settings, prioritizing a higher recall rate is critical.
* **Limitations & Future Improvements**:
    * **Data Size**: The dataset is small (101 instances), limiting the model's capacity to generalize.
    * **Imbalanced Classes**: There are fewer positive depression records than negative records.
    * **Non-Linearity**: Introducing non-linear models like **Decision Trees** or **Random Forests** might better capture the complex socio-academic interactions within the variables.

---

## 🚀 How to Run the Notebook

### 1. Prerequisites
Ensure you have Python installed. Install the necessary dependencies via terminal:
```bash
pip install pandas numpy scikit-learn
