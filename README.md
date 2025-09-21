
# 📉 Customer Churn Prediction – E-commerce App Users – Python & scikit-learn

**Author:** Loc Ha  
**Date:** 2025 August  

---

## 🛠 Tools Used
![Python](https://img.shields.io/badge/Code-Python-blue)
![Pandas](https://img.shields.io/badge/Library-Pandas-yellow)
![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-orange)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-green)
![Seaborn](https://img.shields.io/badge/Library-Seaborn-red)

---

## 📑 Table of Contents
I. [📌 Business Context & Objective](#-business-context--objective)  
II. [📂 Dataset Description & Structure](#-dataset-description--structure)  
III. [⚒️ Main Process](#%EF%B8%8F-main-process)  
IV. [📊 Key Insights & Recommendations](#-key-insights--recommendations)  

---

## 📌 Business Context & Objective

### 🏢 Business Question
How can we **predict and understand churn behavior** among e-commerce app users to design effective retention strategies?  

### 🎯 Objective
- Identify behavioral patterns of churned users.  
- Build a **Machine Learning model** to predict churn with high recall/precision balance.  
- Segment churned users into groups for **personalized marketing campaigns**.  

---

## 📂 Dataset Description & Structure

- **Source**: E-commerce churn dataset (simulated).  
- **Size**: 5,630 rows × 20 columns  
- **Target column**: `Churn` (0 = Active, 1 = Churned)  
- **Missing values**: present in Tenure, DaySinceLastOrder, WarehouseToHome, CouponUsed, etc. (~200–300 each).  
- **ID column**: `CustomerID`  

### 🧩 Data Structure (Simplified)

| Column Name               | Type    | Description                                   |
|---------------------------|---------|-----------------------------------------------|
| CustomerID                | Int     | Unique identifier                             |
| Churn                     | Int     | Target: 1=churned, 0=active                   |
| Tenure                    | Float   | Months of relationship                        |
| PreferredLoginDevice      | Object  | Device used (Mobile/App/Web)                  |
| CityTier                  | Int     | Customer city tier (1–3)                      |
| WarehouseToHome           | Float   | Distance (km)                                 |
| PreferredPaymentMode      | Object  | COD, DebitCard, CreditCard, etc.              |
| Gender                    | Object  | Male/Female                                   |
| HourSpendOnApp            | Float   | Time spent daily on app                       |
| NumberOfDeviceRegistered  | Int     | Registered devices                            |
| PreferedOrderCat          | Object  | Preferred shopping category                   |
| SatisfactionScore         | Int     | Satisfaction score (1–5)                      |
| MaritalStatus             | Object  | Single/Married                                |
| NumberOfAddress           | Int     | Number of saved addresses                     |
| Complain                  | Int     | Whether user complained (0/1)                 |
| OrderAmountHikeFromlastYear | Float | Increase in order amount vs last year         |
| CouponUsed                | Float   | Number of coupons used                        |
| OrderCount                | Float   | Total orders                                  |
| DaySinceLastOrder         | Float   | Days since last order                         |
| CashbackAmount            | Int     | Cashback received                             |

---

## ⚒️ Main Process

1. **Data Preparation**  
   - Handle nulls (imputation).  
   - Encode categorical variables (OneHotEncoder).  
   - Standardize numeric features.  

   <details>
   <summary>📌 View Python code for Data Preparation </summary>

   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder, StandardScaler
   from sklearn.pipeline import Pipeline
   from sklearn.impute import SimpleImputer

   # Separate features and target
   y = df["Churn"]
   X = df.drop(columns=["Churn", "CustomerID"])

   # Numeric & categorical columns
   num_cols = X.select_dtypes(include=["int64","float64"]).columns
   cat_cols = X.select_dtypes(include=["object"]).columns

   # Preprocessor
   numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())])
   categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))])
   preprocessor = ColumnTransformer([("num", numeric, num_cols),
                                     ("cat", categorical, cat_cols)])
   ```
   </details>

2. **Exploratory Data Analysis (EDA)**  
   - **Correlation analysis** (numeric features vs churn).

   <details>
      <summary>📌 View Python code for Numberic features </summary>
   
      ```python
   #Phân tích các biến số học - Numberic features - Correlation
   
   numeric_cols = ['Tenure','SatisfactionScore','DaySinceLastOrder',
                   'OrderCount','CouponUsed','CashbackAmount','HourSpendOnApp']
   
   corrs = {}
   for col in numeric_cols:
       corrs[col] = df[col].corr(df['Churn'])  # Pearson correlation (0/1 với numeric)
   
   print("Correlation với Churn:")
   for k,v in corrs.items():
       print(f"{k}: {v:.3f}")
   
   corr_df = pd.DataFrame.from_dict(corrs, orient='index', columns=['Correlation']).sort_values(by='Correlation')
   
   # Plot bar chart
   plt.figure(figsize=(8,5))
   sns.barplot(x=corr_df.index, y='Correlation', data=corr_df, palette="coolwarm")
   plt.xticks(rotation=45)
   plt.title("Point Biserial Correlation between Churn and Numeric Features")
   plt.axhline(0, color='black', linestyle='--')
   plt.show()
      ```
      <img width="706" height="560" alt="image" src="https://github.com/user-attachments/assets/5cfdee3d-d6e7-4797-81e5-f412c2c3a8b3" />
   
      </details>


  

   - **Chi-square test** (categorical vs churn)

   <details>
    <summary>📌 View Python code for Categorical Features </summary>
   
      ```python
   #Phân tích với biến phân loại - Categorical features - Chi-square test
   
   cat_cols = ['PreferredLoginDevice','PreferredPaymentMode','Gender',
               'MaritalStatus','PreferedOrderCat','Complain']
   
   for col in cat_cols:
   
       # Kiểm định Chi-square - Chi-square tesst 
       crosstab = pd.crosstab(df[col], df['Churn'])
       chi2, p, dof, ex = chi2_contingency(crosstab)
   
       # Tính tỷ lệ churn + số lượng - Calculate churn rate and stats
       summary = df.groupby(col)['Churn'].agg(['mean','count','sum'])
       summary = summary.rename(columns={'mean':'ChurnRate','count':'Total','sum':'Churned'})
       summary = summary.sort_values(by='ChurnRate', ascending=False)
   
       # In bảng số liệu - Print infor table
       print(f"\n=== {col} ===")
       print(summary.round(3))
       print(f"Chi-square test p-value = {p:.6f}")
   
       # Vẽ chart - Visualization
       plt.figure(figsize=(6,4))
       sns.barplot(x=summary.index, y=summary['ChurnRate'], palette="viridis")
       plt.title(f"Churn rate by {col}")
       plt.ylabel("Churn rate")
       plt.xticks(rotation=45)
       plt.show()
      ```
      === PreferredLoginDevice ===
   
      <img width="545" height="455" alt="image" src="https://github.com/user-attachments/assets/1a123b69-997c-49fa-9778-95153eaf8198" />
   
      === PreferredPaymentMode ===
      
      <img width="545" height="472" alt="image" src="https://github.com/user-attachments/assets/46b42435-4e6a-42f7-8222-115a0ab97b6e" />
      
      === Gender ===
      
      <img width="553" height="425" alt="image" src="https://github.com/user-attachments/assets/ba559f89-02d0-46cb-99e6-530a84a60b50" />
      
      === MaritalStatus ===
      
      <img width="545" height="433" alt="image" src="https://github.com/user-attachments/assets/30027f9e-a9ac-4a41-abda-4def11674602" />
      
      === PreferedOrderCat ===
      
      <img width="545" height="486" alt="image" src="https://github.com/user-attachments/assets/3534c883-6b11-4282-8ea8-9130aa6bc0bf" />
      
      === Complain ===
      
      <img width="545" height="395" alt="image" src="https://github.com/user-attachments/assets/7cb6178f-7b23-41d9-8b70-976f93e80eac" />
      
      
      </details>


2. **Modeling**  
   - Algorithms tested: Logistic Regression, Random Forest.  
   - Metrics evaluated: Precision, Recall, F1, ROC-AUC, PR-AUC.  
   - **Best model selected**: Random Forest (balanced).  

   *Placeholder for chart: ROC Curve & Precision-Recall Curve*  
   
<details>
  <summary>📌 View Python code for Modeling (LogReg & RandomForest, balanced)</summary>

```python
# Modeling: Logistic Regression & Random Forest (balanced) + ROC/PR evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Target & features
y = df["Churn"].astype(int)
X = df.drop(columns=["Churn", "CustomerID"])

# Column types
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Preprocessing
numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])

pre = ColumnTransformer([("num", numeric, num_cols),
                         ("cat", categorical, cat_cols)])

# Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Models
#Logistic Regression - Hồi quy tuyến tính
pipe_lr = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
])

param_lr = {
    "clf__C": np.logspace(-3, 2, 30)  # 1e-3 → 1e2
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rs_lr = RandomizedSearchCV(
    pipe_lr, param_distributions=param_lr, n_iter=25,
    scoring="roc_auc", cv=cv, random_state=42, n_jobs=-1, verbose=1
)
rs_lr.fit(X_train, y_train)
print("Best ROC-AUC (CV):", rs_lr.best_score_)
print("Best params:", rs_lr.best_params_)

#Random Forest
pipe_rf = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        class_weight="balanced", n_jobs=-1, random_state=42
    ))
])

param_rf = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 8, 12],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features": ["sqrt"]
}

rs_rf = RandomizedSearchCV(
    pipe_rf, param_distributions=param_rf, n_iter=10,
    scoring="roc_auc", cv=3, random_state=42, n_jobs=-1, verbose=1
)
rs_rf.fit(X_train, y_train)
print("Best ROC-AUC (CV):", rs_rf.best_score_)
print("Best params:", rs_rf.best_params_)

# Chọn mô hình báo cáo (theo yêu cầu: Random Forest)
best_name = "rf"
print("\nBest model selected for report:", best_name)

# Vẽ ROC & PR cho model best
best_pipe = Pipeline([("pre", pre), ("clf", models[best_name])])
best_pipe.fit(X_tr, y_tr)
best_prob = best_pipe.predict_proba(X_te)[:, 1]

RocCurveDisplay.from_predictions(y_te, best_prob)
plt.title(f"ROC Curve - {best_name.upper()}")
plt.show()

PrecisionRecallDisplay.from_predictions(y_te, best_prob)
plt.title(f"Precision-Recall Curve - {best_name.upper()}")
plt.show()
```
</details>
3. **Segmentation (Clustering)**  
   - KMeans clustering applied on churned users.  
   - Elbow method → k=4 optimal clusters.  

   *Placeholder for chart: Cluster Distribution (k=4)*  

---

## 📊 Key Insights & Recommendations

### 💡 Insights from EDA
- **Tenure**: Lower tenure → higher churn → retention strategy for new users.  
- **DaySinceLastOrder**: More inactive days → higher churn → reactivation campaigns needed.  
- **CashbackAmount**: More cashback → less churn → cashback effective as retention lever.  
- **COD Payment**: COD users churn more → promote digital payments.  
- **Complain**: Complaints strongly linked to churn → need complaint resolution focus.  
- **iPhone users**: Highest churn rate → investigate reasons (UX, compatibility, service).  

### 🔎 Segmentation Results (Clusters)
- **Cluster 0**: Long-tenure, high cashback, low activity → loyalty rewards & small-order coupons.  
- **Cluster 1**: High order count, high coupon use, but low satisfaction → improve delivery & service quality.  
- **Cluster 2**: New users, high app usage, high satisfaction → encourage repeat orders with welcome coupons.  
- **Cluster 3**: Very new, low activity → onboarding campaigns, free shipping & first-purchase vouchers.  

### 📝 Recommendations
1. Retain **long-tenure users** with loyalty programs.  
2. Target **COD users & complainers** with education + resolution.  
3. Encourage **digital payments** & cashback-based campaigns.  
4. Reactivate inactive users via personalized promotions.  
5. Segment-specific promotions (based on cluster analysis).  

---
