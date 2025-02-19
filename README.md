# Loan Prediction Using Machine Learning

## Project Overview
This project aims to predict **loan approval status** using machine learning models. The dataset consists of various applicant details, including income, loan amount, credit history, and demographic information. After evaluating multiple models, **Random Forest** was found to perform best, achieving an accuracy of **0.88**.

## Dataset Information
The dataset contains the following key features:
- **Applicant Income & Coapplicant Income**: Combined to form "Total_Income."
- **Credit History**: A significant factor in loan approval.
- **Marital Status, Education, and Self-Employment**: Categorical variables affecting loan eligibility.
- **Loan Amount & Loan Term**: The requested loan amount and repayment duration.
- **Property Area**: Encoded as Urban, Semi-Urban, and Rural.
- **Loan Status (Target Variable)**: 1 for approved, 0 for rejected.

## Preprocessing Steps
- **Handling Missing Values:** Median imputation for numerical columns; mode imputation for categorical columns.
- **Feature Engineering:** Created "Total_Income" and dropped "ApplicantIncome" & "CoapplicantIncome."
- **Encoding:** Binary mapping for categorical variables and one-hot encoding for multi-class features.
- **Scaling (if needed):** Standardization applied for models requiring normalized input.

## Model Selection & Evaluation
We tested multiple models:
### Logistic Regression
- **Accuracy:** Lowest
- **Less suitable for this dataset**

### Random Forest (Best Model)
- **Accuracy:** 0.88
- **Hyperparameter Tuning:** GridSearchCV for optimization
- **Strengths:** Handles missing values, categorical features, and feature importance well.

### XGBoost (Alternative Model)
- **Accuracy:** Lower than Random Forest in this dataset
- **More sensitive to hyperparameter tuning**

## Installation & Usage
### Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Run the Notebook
Open the `loan-prediction.ipynb` file and execute the cells step by step to preprocess data, train models, and evaluate performance.

## Code Workflow
1. **Load Data**: The dataset is read using Pandas.
2. **Preprocessing**:
   - Missing values are handled (median for numerical, mode for categorical).
   - Categorical variables are encoded using binary mapping and one-hot encoding.
   - Feature engineering is applied (e.g., "Total_Income" created).
3. **Feature Selection**:
   - Features are analyzed for importance.
   - Unnecessary features are removed to improve model efficiency.
4. **Model Training**:
   - Models like Logistic Regression, Random Forest and XGBoost are trained and compared.
   - GridSearchCV is used for hyperparameter tuning.
5. **Model Evaluation**:
   - Accuracy and other performance metrics are calculated.
   - The confusion matrix is analyzed to determine classification performance.

## Future Enhancements
- **More Feature Engineering** (e.g., loan-to-income ratio)
- **Ensemble Learning** (Stacking multiple models for better results)
- **Hyperparameter Optimization** (Further tuning using Bayesian Optimization)

---
**Dataset Source**: Kaggle

