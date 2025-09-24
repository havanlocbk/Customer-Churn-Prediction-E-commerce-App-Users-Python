
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

1. **Exploratory Data Analysis (EDA)**  
   - Load dataset from Google Sheets (CSV format).  
   - Inspect columns, datatypes, and summary statistics.  
   - Check class distribution of target `Churn`.  
   - Identify missing values.  

   <details>
   <summary>📌 View Python code for EDA </summary>

   ```python
   # Load libraries
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   from scipy.stats import chi2_contingency, ttest_ind

   # Load dataset from Google Sheets
   file_id = "1yxgr0Qj3TiXRehYa0PED1t4zIga9gdY5"
   url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
   df = pd.read_csv(url)

   # Inspect data
   print(df.head())
   print(df.columns)
   df.info()
   df.describe()

   # Target distribution
   print(df['Churn'].value_counts(normalize=True))

   # Missing values
   print(df.isnull().sum())

2. **Exploratory Data Analysis – Numeric Features Correlation**  
   - Select numeric variables: `Tenure`, `SatisfactionScore`, `DaySinceLastOrder`, `OrderCount`, `CouponUsed`, `CashbackAmount`, `HourSpendOnApp`.  
   - Calculate Pearson correlation between numeric features and `Churn`.  
   - Visualize correlations with bar chart.  

   <details>
   <summary>📌 View Python code for Numeric Correlation </summary>

   ```python
   # Correlation analysis
   numeric_cols = ['Tenure','SatisfactionScore','DaySinceLastOrder',
                   'OrderCount','CouponUsed','CashbackAmount','HourSpendOnApp']

   corrs = {}
   for col in numeric_cols:
       corrs[col] = df[col].corr(df['Churn'])  # Pearson correlation (0/1 with numeric)

   print("Correlation with Churn:")
   for k,v in corrs.items():
       print(f"{k}: {v:.3f}")

   # Visualization
   corr_df = pd.DataFrame.from_dict(corrs, orient='index', columns=['Correlation']).sort_values(by='Correlation')

   plt.figure(figsize=(8,5))
   sns.barplot(x=corr_df.index, y='Correlation', data=corr_df, palette="coolwarm")
   plt.xticks(rotation=45)
   plt.title("Point Biserial Correlation between Churn and Numeric Features")
   plt.axhline(0, color='black', linestyle='--')
   plt.show()
   ```
   </details>
      
   ### 📍Output (correlation values):
   
   - Tenure: -0.349
   
   - SatisfactionScore: 0.105
   
   - DaySinceLastOrder: -0.161
   
   - OrderCount: -0.029
   
   - CouponUsed: -0.008
   
   - CashbackAmount: -0.154
   
   - HourSpendOnApp: 0.019
   
   ### 📊 Visualization
   Bar chart showing correlations  
   <img width="706" height="560" alt="image" src="https://github.com/user-attachments/assets/bf33d456-16cb-496a-b626-98df4bc13b5c" />

      **Interpretation:**  
   - Longer **Tenure** and higher **CashbackAmount** strongly reduce churn.  
   - More **recent orders** also reduce churn (DaySinceLastOrder negative).  
   - **SatisfactionScore** shows a surprising positive correlation → may indicate misalignment between score and true loyalty.  
   - **OrderCount**, **CouponUsed**, **HourSpendOnApp** have near-zero impact on churn.  

### 3. Categorical Features – Chi-square Test

<details>
  <summary>📌 View Python code</summary>

```python
# Phân tích với biến phân loại (Categorical features) - Chi-square test

cat_cols = ['PreferredLoginDevice','PreferredPaymentMode','Gender',
            'MaritalStatus','PreferedOrderCat','Complain']

# 1. Chuẩn hóa text trong các cột phân loại - Standardize text
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.title()  # đồng nhất viết hoa chữ cái đầu

# 2. Mapping thủ công nếu có giá trị cần gộp - Mapping values manually
replace_dict = {
    'PreferredLoginDevice': {
        'Mobile Phone': 'Phone',
        'Phone': 'Phone'
    },
    'PreferredPaymentMode': {
        'Debit Card': 'Card',
        'Credit Card': 'Card',
        'Cc': 'Card',
        'Cash On Delivery':'COD',
        'Cod':'COD'
    },
}

df = df.replace(replace_dict)  # Replace và chuẩn hóa giá trị

# 3. Phân tích và trực quan hóa
for col in cat_cols:

    # Kiểm định Chi-square
    crosstab = pd.crosstab(df[col], df['Churn'])
    chi2, p, dof, ex = chi2_contingency(crosstab)

    # Tính churn rate + số lượng
    summary = df.groupby(col)['Churn'].agg(['mean','count','sum'])
    summary = summary.rename(columns={'mean':'ChurnRate','count':'Total','sum':'Churned'})
    summary = summary.sort_values(by='ChurnRate', ascending=False)

    # In bảng kết quả
    print(f"\n=== {col} ===")
    print(summary.round(3))
    print(f"Chi-square test p-value = {p:.6f}")

    # Vẽ chart
    plt.figure(figsize=(6,4))
    sns.barplot(x=summary.index, y=summary['ChurnRate'], palette="viridis")
    plt.title(f"Churn rate by {col}")
    plt.ylabel("Churn rate")
    plt.xticks(rotation=45)
    plt.show()
```
</details>

   === PreferredLoginDevice ===
                         ChurnRate  Total  Churned
   PreferredLoginDevice                           
   Computer                  0.198   1634      324
   Phone                     0.156   3996      624
   
   Chi-square test p-value = 0.000148
   <img width="553" height="438" alt="image" src="https://github.com/user-attachments/assets/c626a3bc-972a-4d6e-96c1-d187f2320416" />

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
