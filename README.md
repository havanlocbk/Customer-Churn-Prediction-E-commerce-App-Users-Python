
# 📉 Customer Churn Prediction – E-commerce App Users – Python

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
   <summary>📌 View Python code</summary>

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
   - **Chi-square test** (categorical vs churn).  

   *Placeholder for chart: Correlation Heatmap*  
   *Placeholder for chart: Churn Rate by Category*  

3. **Modeling**  
   - Algorithms tested: Logistic Regression, Random Forest.  
   - Metrics evaluated: Precision, Recall, F1, ROC-AUC, PR-AUC.  
   - **Best model selected**: Random Forest (balanced).  

   *Placeholder for chart: ROC Curve & Precision-Recall Curve*  

4. **Segmentation (Clustering)**  
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
