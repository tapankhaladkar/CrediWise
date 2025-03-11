# CrediWise
An AI-powered system for intelligent credit score classification and risk assessment.
# Credit Score Classification Project

## Project Overview
This project focuses on building a **Machine Learning model** to classify individuals into different **credit score brackets** based on their financial data. The dataset consists of **100,000 records** with **28 financial attributes**, and multiple classification models were implemented to improve prediction accuracy.

## Problem Statement
As a **Data Scientist** at a **global finance company**, the objective is to build an **intelligent system** that classifies people into **credit score brackets**, reducing **manual efforts** and improving financial decision-making.

## Dataset Information
The dataset contains **28 financial attributes** related to customers, including:

- **Demographics:** `Age`, `Occupation`, `Customer_ID`
- **Financial Status:** `Annual_Income`, `Monthly_Inhand_Salary`, `Num_Bank_Accounts`
- **Loan Information:** `Num_of_Loan`, `Type_of_Loan`, `Outstanding_Debt`
- **Credit Behavior:** `Num_of_Delayed_Payment`, `Credit_Utilization_Ratio`, `Credit_Mix`
- **Repayment Patterns:** `Payment_of_Min_Amount`, `Total_EMI_per_month`
- **Target Variable:** `Credit_Score` (classification category)

## Technologies & Frameworks Used
- **Programming Language:** Python
- **Machine Learning Libraries:** Scikit-learn, Pandas, NumPy
- **Visualization Tools:** Matplotlib, Seaborn
- **Data Processing:** One-Hot Encoding, Feature Selection (VIF)
- **Model Optimization:** Hyperparameter Tuning (GridSearchCV)

## Project Workflow
### 1. Data Preprocessing
- Removed unnecessary columns.
- Replaced **special characters** and converted data into appropriate types.
- Handled **missing values** using **Forward & Backward Filling** methods.
- **Outlier detection & removal** (focused on `Age` column).

### 2. Feature Engineering
- Applied **One-Hot Encoding** to categorical variables.
- Selected features using **Variance Inflation Factor (VIF)** ensuring **VIF < 5**.

### 3. Model Implementation & Results
Several **Machine Learning Models** were trained and tested:

| **Model**             | **Accuracy**  |
|-----------------------|--------------|
| **Logistic Regression**  | 61.8%   |
| **Decision Tree**        | 69.7%   |
| **Decision Tree (Tuned)** | **70.93%** |
| **Random Forest**       | **79.7%** |

### 4. Hyperparameter Tuning
- **GridSearchCV** was used to optimize the **Decision Tree model**, improving accuracy from **69.7% to 70.93%**.

## How to Run the Project
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Credit-Score-Classification.git
cd Credit-Score-Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook
```
- Open `Credit_Score.ipynb` and execute the cells.

## Future Improvements
- Implement **XGBoost** and **Neural Networks** for better accuracy.
- Improve feature selection techniques using **Recursive Feature Elimination (RFE)**.
- Deploy the model using **Flask or FastAPI** for real-world applications.

---

