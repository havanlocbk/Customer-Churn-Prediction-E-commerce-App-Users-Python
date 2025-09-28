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
1. [📌 Business Context & Objective](#1-business-context--objective)  
2. [📂 Dataset Description & Structure](#2-dataset-description--structure)  
3. [⚒️ Main Process](#3-main-process)  
  3.1. Exploratory Data Analysis (EDA)  
  3.2. Modeling  
  3.3. Segmentation (Clustering)  
4. [📊 Key Insights & Recommendations](#4-key-insights--recommendations)  

---

## 1. 📌 Business Context & Objective  

### 🏢 Business Question  
How can we **predict and understand churn behavior** among e-commerce app users to design effective retention strategies?  

### 🎯 Objective  
- Identify behavioral patterns of churned users.  
- Build a **Machine Learning model** to predict churn with high recall/precision balance.  
- Segment churned users into groups for **personalized marketing campaigns**.  

---

## 2. 📂 Dataset Description & Structure  

- **Source**: E-commerce churn dataset (simulated).  
- **Size**: 5,630 rows × 20 columns  
- **Target column**: `Churn` (0 = Active, 1 = Churned)  
- **Missing values**: present in Tenure, DaySinceLastOrder, WarehouseToHome, CouponUsed, etc. (~200–300 each).  
- **ID column**: `CustomerID`  

### 🧩 Data Structure (Simplified)  

| Column Name                  | Type    | Description                                   |
|------------------------------|---------|-----------------------------------------------|
| CustomerID                   | Int     | Unique identifier                             |
| Churn                        | Int     | Target: 1=churned, 0=active                   |
| Tenure                       | Float   | Months of relationship                        |
| PreferredLoginDevice         | Object  | Device used (Mobile/App/Web)                  |
| CityTier                     | Int     | Customer city tier (1–3)                      |
| WarehouseToHome              | Float   | Distance (km)                                 |
| PreferredPaymentMode         | Object  | COD, DebitCard, CreditCard, etc.              |
| Gender                       | Object  | Male/Female                                   |
| HourSpendOnApp               | Float   | Time spent daily on app                       |
| NumberOfDeviceRegistered     | Int     | Registered devices                            |
| PreferedOrderCat             | Object  | Preferred shopping category                   |
| SatisfactionScore            | Int     | Satisfaction score (1–5)                      |
| MaritalStatus                | Object  | Single/Married                                |
| NumberOfAddress              | Int     | Number of saved addresses                     |
| Complain                     | Int     | Whether user complained (0/1)                 |
| OrderAmountHikeFromlastYear  | Float   | Increase in order amount vs last year         |
| CouponUsed                   | Float   | Number of coupons used                        |
| OrderCount                   | Float   | Total orders                                  |
| DaySinceLastOrder            | Float   | Days since last order                         |
| CashbackAmount               | Int     | Cashback received                             |

---

## 3. ⚒️ Main Process  

### 3.1 Exploratory Data Analysis (EDA)  

#### 🔹 Numeric Features Correlation  

<details>
<summary>📌 View Python code</summary>

```python
# Correlation analysis
numeric_cols = ['Tenure','SatisfactionScore','DaySinceLastOrder',
                'OrderCount','CouponUsed','CashbackAmount','HourSpendOnApp']

corrs = {}
for col in numeric_cols:
    corrs[col] = df[col].corr(df['Churn'])

corr_df = pd.DataFrame.from_dict(corrs, orient='index', columns=['Correlation']).sort_values(by='Correlation')
print(corr_df)
```
</details>

*Placeholder for chart: Correlation bar chart (numeric features vs churn)*  

📍 **Key findings**:  
- Longer **Tenure** and higher **CashbackAmount** strongly reduce churn.  
- More **recent orders** also reduce churn (DaySinceLastOrder negative).  
- **SatisfactionScore** surprisingly shows positive correlation → misalignment between survey score & true loyalty.  
- **OrderCount**, **CouponUsed**, **HourSpendOnApp** have near-zero impact.  

---

#### 🔹 Categorical Features (Chi-square Test)  

<details>
<summary>📌 View Python code</summary>

```python
from scipy.stats import chi2_contingency

cat_cols = ['PreferredLoginDevice','PreferredPaymentMode','Gender',
            'MaritalStatus','PreferedOrderCat','Complain']

for col in cat_cols:
    crosstab = pd.crosstab(df[col], df['Churn'])
    chi2, p, dof, ex = chi2_contingency(crosstab)
    print(f"{col} | p-value = {p:.6f}")
```
</details>

*Placeholder for chart: Churn rate by categorical features*  

📍 **Key findings**:  
- **COD Payment** users churn more.  
- **Complaints** highly correlated with churn.  
- **PreferredLoginDevice = Computer/iPhone** → higher churn rates.  

---

### 3.2 Modeling  

#### 🔹 Algorithms Tested  
- Logistic Regression (baseline model)  
- Random Forest (balanced) → **selected**  

#### 🔹 Metrics Evaluated  
- Precision, Recall, F1-score  
- ROC-AUC, PR-AUC  
- Threshold tuning  

<details>
<summary>📌 View Python code for Modeling</summary>

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight='balanced', random_state=42
)
rf.fit(X_train, y_train)
y_proba_rf = rf.predict_proba(X_test)[:,1]

# Apply chosen threshold
threshold = 0.52
y_pred_rf = (y_proba_rf >= threshold).astype(int)
```
</details>

**Threshold tuning results:**  
- Optimal F1 threshold = 0.350 → Precision=0.881 | Recall=0.937  
- Recall ≥ 0.80 threshold = 0.520 → Precision=0.921 | Recall=0.800  

✅ **Chosen threshold = 0.520** → balanced Precision & Recall.  

**Confusion Matrix & Classification Report (Threshold=0.520)**  

```
[[923  13]
 [ 38 152]]
              precision    recall  f1-score   support

           0      0.960     0.986     0.973       936
           1      0.921     0.800     0.856       190

    accuracy                          0.955      1126
   macro avg      0.941     0.893     0.915      1126
weighted avg      0.954     0.955     0.953      1126
```

*Placeholder for chart: Confusion Matrix (Random Forest, threshold=0.52)*  
*Placeholder for chart: ROC Curve*  
*Placeholder for chart: Precision-Recall Curve*  

---

### 3.3 Segmentation (Clustering)  

#### 🔹 Approach  
- Apply KMeans clustering on churned users.  
- Elbow method → optimal k=4.  
- Analyze segment characteristics.  

<details>
<summary>📌 View Python code for Clustering</summary>

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df_churned = df[df['Churn']==1].copy()
features = ['Tenure','SatisfactionScore','OrderCount','CouponUsed','CashbackAmount','HourSpendOnApp']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_churned[features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_churned['Cluster'] = kmeans.fit_predict(X_scaled)
```
</details>

*Placeholder for chart: Elbow Method (SSE vs k)*  
*Placeholder for chart: Cluster Distribution (k=4)*  

📍 **Cluster insights**:  
- **Cluster 0**: Long-tenure, high cashback, low activity → loyalty rewards & small-order coupons.  
- **Cluster 1**: High order count, heavy coupon use, but low satisfaction → improve delivery & service quality.  
- **Cluster 2**: New users, high app usage, high satisfaction → encourage repeat orders with welcome offers.  
- **Cluster 3**: Very new, low activity → onboarding campaigns, free shipping & first-purchase vouchers.  

---

## 4. 📊 Key Insights & Recommendations  

### 💡 Insights from EDA  
- Tenure ↓ → churn ↑ (new users at risk).  
- Inactive days ↑ → churn ↑ (reactivation campaigns needed).  
- Cashback ↑ → churn ↓ (effective lever).  
- COD payment & complaints strongly linked to churn.  
- iPhone/computer users churn more.  

---

### 🔎 Segmentation Results  
- **Cluster 0**: Long-tenure, cashback → loyalty rewards.  
- **Cluster 1**: High orders + low satisfaction → service improvement.  
- **Cluster 2**: New & engaged users → welcome offers.  
- **Cluster 3**: Very new, low activity → onboarding campaigns.  

---

### 📝 Recommendations  
1. 🎯 Focus on **new user retention** via onboarding & welcome offers.  
2. 💳 Promote **digital payments** to reduce COD churn.  
3. 🎁 Strengthen **cashback & loyalty** for long-tenure users.  
4. ⚡ Reactivate inactive users with **personalized promotions**.  
5. 🛠 Improve **customer service** to reduce churn from complaints.  
6. 👥 Deploy **cluster-based marketing** for targeted retention.  
