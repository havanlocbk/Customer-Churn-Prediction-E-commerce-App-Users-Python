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

---

## 3. ⚒️ Main Process

### 3.1 Exploratory Data Analysis (EDA)

#### 🔹 Load dataset & initial inspection  
👉 **Purpose:** Load raw dataset, inspect structure, summary statistics, and missing values.  

<details>
<summary>📌 View Python code</summary>

```python
#Load file và phân tích EDA - Loading and EDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind

file_id = "1yxgr0Qj3TiXRehYa0PED1t4zIga9gdY5"
url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"

df = pd.read_csv(url)
print(df.head())
print(df.columns)
df.info()
df.describe()
df['Churn'].value_counts(normalize=True)
print(df.isnull().sum())
```
</details>




---

#### 🔹 Numeric Features – Correlation  
👉 **Purpose:** Calculate Pearson correlation between numeric features and churn to identify key drivers.  

<details>
<summary>📌 View Python code</summary>

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

#Phân tích các biến số học - Numberic features - Visualization
corr_df = pd.DataFrame.from_dict(corrs, orient='index', columns=['Correlation']).sort_values(by='Correlation')

# Plot bar chart
plt.figure(figsize=(8,5))
sns.barplot(x=corr_df.index, y='Correlation', data=corr_df, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Point Biserial Correlation between Churn and Numeric Features")
plt.axhline(0, color='black', linestyle='--')
plt.show()
```
</details>

  Tenure: -0.349
  SatisfactionScore: 0.105
  DaySinceLastOrder: -0.161
  OrderCount: -0.029
  CouponUsed: -0.008
  CashbackAmount: -0.154
  HourSpendOnApp: 0.019


📍 **Key findings:**  
- Tenure ↓ → churn ↑
- CashbackAmount ↑ → churn ↓
- DaySinceLastOrder negative correlation with churn
- SatisfactionScore unexpected positive correlation  


<img width="706" height="560" alt="image" src="https://github.com/user-attachments/assets/92b8fa7d-562e-4b55-94b0-e2339c96c847" />

---

#### 🔹 Categorical Features – Chi-square test  
👉 **Purpose:** Perform chi-square test on categorical variables to detect significant associations with churn.  

<details>
<summary>📌 View Python code</summary>

```python
#-------------------------

# Phân tích với biến phân loại (Categorical features) - Chi-square test

cat_cols = ['PreferredLoginDevice','PreferredPaymentMode','Gender',
            'MaritalStatus','PreferedOrderCat','Complain']

# 1. Chuẩn hóa text trong các cột phân loại - Standadize text
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

df = df.replace(replace_dict) #Replace gom về 1 giá trị - Replace and standadize values

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

  <img width="545" height="429" alt="image" src="https://github.com/user-attachments/assets/018c2f6e-905d-4fbf-8ed8-cc5d3cb90fcc" />

  <img width="545" height="395" alt="image" src="https://github.com/user-attachments/assets/e99a2d4c-c93a-41f5-844b-652c1d0cca08" />

  <img width="545" height="486" alt="image" src="https://github.com/user-attachments/assets/c500d8e3-1e12-4e27-98cb-def69f071357" />



📍 **Key findings:**  
- COD payment method linked to higher churn
- Complaints strongly associated with churn
- Certain devices show higher churn rates  


---

### 3.2 Modeling

#### 🔹 Import libraries  
👉 **Purpose:** Import neccessary libraries for Modelling.  

<details>
<summary>📌 View Python code</summary>

```python
# Setup & đọc dữ liệu - Setup and import libraries

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, confusion_matrix)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
```
</details>



---

#### 🔹 Standadize and split dataset  
👉 **Purpose:** Split dataset into training and testing sets with stratification on churn variable

<details>
<summary>📌 View Python code</summary>

```python
# Chuẩn hóa tên cột - Columns standadize
df.columns = df.columns.str.strip()

# Chuẩn hóa target Churn về 0/1 - Target & features
if df["Churn"].dtype == object:
    df["Churn"] = (df["Churn"].astype(str).str.strip()
                   .str.lower()
                   .map({"yes":1, "1":1, "true":1, "no":0, "0":0, "false":0})).astype(int)

# Tách X, y - Separate features and target
y = df["Churn"].astype(int)
X = df.drop(columns=["Churn"])

# Phân loại feature type - Column types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("Số mẫu:", len(df))
print("Tỷ lệ churn:", y.mean().round(4))
print("Số numeric:", len(num_cols), "| Số categorical:", len(cat_cols))
```
</details>

<details>
<summary>📌 View Python code</summary>
  
```python
#Pre-processing - tiền xử lý

num_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_tf, num_cols),
    ("cat", cat_tf, cat_cols)
])
```
</details>

<details>
<summary>📌 View Python code</summary>
  
```python
#Train, Valid, Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pos = y_train.sum(); neg = len(y_train) - pos
scale_pos_weight = (neg / pos) if pos > 0 else 1.0
scale_pos_weight
```
</details>

---

#### 🔹 Model Evaluation  
👉 **Purpose:** Evaluate models using classification report, confusion matrix, and ROC-AUC.  

<details>
<summary>📌 View Python code</summary>
  
  ```python
  #-------------------------
  
  # Baseline Models (chưa tuning)
  from sklearn.metrics import roc_auc_score, average_precision_score
  
  # Logistic Regression (baseline)
  pipe_lr = Pipeline([
      ("prep", preprocessor),
      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
  ])
  pipe_lr.fit(X_train, y_train)
  
  y_pred_lr = pipe_lr.predict(X_test)
  y_proba_lr = pipe_lr.predict_proba(X_test)[:,1]
  
  print("Logistic Regression (Baseline)")
  print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr).round(4))
  print("PR-AUC:", average_precision_score(y_test, y_proba_lr).round(4))
  
  # Random Forest (baseline)
  pipe_rf = Pipeline([
      ("prep", preprocessor),
      ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42))
  ])
  pipe_rf.fit(X_train, y_train)
  
  y_pred_rf = pipe_rf.predict(X_test)
  y_proba_rf = pipe_rf.predict_proba(X_test)[:,1]
  
  print("Random Forest (Baseline)")
  print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf).round(4))
  print("PR-AUC:", average_precision_score(y_test, y_proba_rf).round(4))
  ```
</details>

👉👉👉

Logistic Regression (Baseline)
ROC-AUC: 0.8873
PR-AUC: 0.6671
Random Forest (Baseline)
ROC-AUC: 0.9876
PR-AUC: 0.9538

📍 **Key findings:**  
- Random Forest outperforms Logistic Regression on recall and F1.  


---

#### 🔹 Threshold tuning – F1 optimization  
👉 **Purpose:** Tune decision threshold to maximize F1 score.  

<details>
<summary>📌 View Python code</summary>
  
  ```python
    from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
    import numpy as np
    
    # Xác suất dự đoán churn từ mô hình RF
    y_proba = pipe_rf.predict_proba(X_test)[:,1]   # best_model = RF đã fit
    y_true = y_test
    
    # Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)  # khép kín 1.0
    
    # Tính F1 cho từng threshold
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    idx_f1 = np.nanargmax(f1s)
    
    thr_f1 = thresholds[idx_f1]
    print(f"Ngưỡng tối ưu theo F1 = {thr_f1:.3f} | Precision={precisions[idx_f1]:.3f} | Recall={recalls[idx_f1]:.3f}")
    
    # Chọn ngưỡng để đạt Recall ≥ 0.80
    target_recall = 0.80
    mask = recalls >= target_recall
    if mask.any():
        idx_rec = np.argmax(precisions[mask])  # chọn precision cao nhất trong số recall ≥ 0.8
        idx_rec = np.where(mask)[0][idx_rec]
        thr_rec = thresholds[idx_rec]
        print(f"Ngưỡng đạt Recall ≥ {target_recall}: {thr_rec:.3f} | Precision={precisions[idx_rec]:.3f} | Recall={recalls[idx_rec]:.3f}")
    else:
        print("Không đạt được Recall ≥ 0.80 với bất kỳ ngưỡng nào.")
    
    # Đánh giá confusion matrix tại threshold tối ưu F1
    y_pred_f1 = (y_proba >= thr_f1).astype(int)
    print("\n=== Kết quả với threshold tối ưu F1 ===")
    print(confusion_matrix(y_true, y_pred_f1))
    print(classification_report(y_true, y_pred_f1, digits=3))
    
    # Đánh giá tại threshold Recall≥0.80
    if mask.any():
        y_pred_rec = (y_proba >= thr_rec).astype(int)
        print("\n=== Kết quả với threshold Recall≥0.80 ===")
        print(confusion_matrix(y_true, y_pred_rec))
        print(classification_report(y_true, y_pred_rec, digits=3))
  ```
</details>

👉 **Purpose:** Select threshold ensuring Recall ≥ 0.8 for churn detection.  

<details>
<summary>📌 View Python code</summary>

  ```python
# Lưu kết quả
results = pd.DataFrame({
    "CustomerID": X_test["CustomerID"].values if "CustomerID" in X_test.columns else range(len(X_test)),
    "y_true": y_test.values,
    "y_proba": y_proba,
    "y_pred_F1": y_pred_f1,
    "y_pred_Recall80": (y_proba >= 0.520).astype(int)  #y_proba Ngưỡng đạt Recall ≥ 0.8: 0.520 ưu tiên cân bằng giữa Precision & Recall 
})
results.head()
  ```

</details>
---

📍 **Key findings:**  
- Threshold ~0.52 → Recall ~0.80, Precision ~0.92.
- Chosen threshold = 0.52 (balanced trade-off).   

---
#### 🔹 Threshold tuning – Recall ≥ 0.8  
👉 **Purpose:** Select threshold ensuring Recall ≥ 0.8 for churn detection.

<details>
<summary>📌 View Python code</summary>

```python
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, roc_auc_score, average_precision_score
import numpy as np

# Xác suất dự đoán churn từ mô hình RF
y_proba = pipe_rf.predict_proba(X_test)[:,1]   # best_model = RF đã fit
y_true = y_test

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
thresholds = np.append(thresholds, 1.0)  # khép kín 1.0

# Tính F1 cho từng threshold
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
idx_f1 = np.nanargmax(f1s)

thr_f1 = thresholds[idx_f1]
print(f"Ngưỡng tối ưu theo F1 = {thr_f1:.3f} | Precision={precisions[idx_f1]:.3f} | Recall={recalls[idx_f1]:.3f}")

# Chọn ngưỡng để đạt Recall ≥ 0.80
target_recall = 0.80
mask = recalls >= target_recall
if mask.any():
    idx_rec = np.argmax(precisions[mask])  # chọn precision cao nhất trong số recall ≥ 0.8
    idx_rec = np.where(mask)[0][idx_rec]
    thr_rec = thresholds[idx_rec]
    print(f"Ngưỡng đạt Recall ≥ {target_recall}: {thr_rec:.3f} | Precision={precisions[idx_rec]:.3f} | Recall={recalls[idx_rec]:.3f}")
else:
    print("Không đạt được Recall ≥ 0.80 với bất kỳ ngưỡng nào.")

# Đánh giá confusion matrix tại threshold tối ưu F1
y_pred_f1 = (y_proba >= thr_f1).astype(int)
print("\n=== Kết quả với threshold tối ưu F1 ===")
print(confusion_matrix(y_true, y_pred_f1))
print(classification_report(y_true, y_pred_f1, digits=3))

# Đánh giá tại threshold Recall≥0.80
if mask.any():
    y_pred_rec = (y_proba >= thr_rec).astype(int)
    print("\n=== Kết quả với threshold Recall≥0.80 ===")
    print(confusion_matrix(y_true, y_pred_rec))
    print(classification_report(y_true, y_pred_rec, digits=3))
```
</details>

<details>
<summary>📌 View Python code</summary>
# Lưu kết quả
results = pd.DataFrame({
    "CustomerID": X_test["CustomerID"].values if "CustomerID" in X_test.columns else range(len(X_test)),
    "y_true": y_test.values,
    "y_proba": y_proba,
    "y_pred_F1": y_pred_f1,
    "y_pred_Recall80": (y_proba >= 0.520).astype(int)  #y_proba Ngưỡng đạt Recall ≥ 0.8: 0.520 ưu tiên cân bằng giữa Precision & Recall 
})
results.head()
```
</details>

---

### 3.3 Segmentation (Clustering)

#### 🔹 Prepare churned subset  &  Feature selection & scaling  
👉 **Purpose:** Filter dataset for churned users only for segmentation analysis. And select features for clustering and apply standard scaling.  

<details>
<summary>📌 View Python code</summary>

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Lọc churn users
churn_users = df[df["Churn"] == 1].copy()

# Chọn một số biến hành vi quan trọng để phân cụm
features = ["Tenure", "DaySinceLastOrder", "OrderCount", "CouponUsed",
            "CashbackAmount", "SatisfactionScore", "HourSpendOnApp"]

# Xử lý missing nếu có
X_cluster = churn_users[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X_cluster)

# Chọn số cluster bằng Elbow method
wcss = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 7), wcss, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
```

</details> 

---

#### 🔹 Elbow method  
👉 **Purpose:** Use SSE to determine optimal number of clusters.  

<details>
<summary>📌 View Python code</summary>

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Lọc churn users
churn_users = df[df["Churn"]==1].copy()

# Chọn một số biến hành vi quan trọng để phân cụm
features = ["Tenure","DaySinceLastOrder","OrderCount","CouponUsed",
            "CashbackAmount","SatisfactionScore","HourSpendOnApp"]

# Xử lý missing nếu có
X_cluster = churn_users[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X_cluster)

# Chọn số cluster bằng Elbow method
wcss = []
for k in range(2,7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(2,7), wcss, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
```
</details>

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/3e1f5dc4-d515-40a3-a5da-c9a4c9401450" />


📍 **Key findings:**  
- Elbow suggests k=4 clusters.  

---

#### 🔹 KMeans clustering  
👉 **Purpose:** Fit final KMeans model (k=4) and assign cluster labels to churned users.  

<details>
<summary>📌 View Python code</summary>

```python
# Chọn k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
churn_users["Cluster"] = kmeans.fit_predict(X_scaled)

# Xem đặc trưng trung bình mỗi nhóm
cluster_summary = churn_users.groupby("Cluster")[features].mean().round(2)
print(cluster_summary)
```
</details>

<img width="710" height="268" alt="image" src="https://github.com/user-attachments/assets/686ad50f-6b8f-40bd-825a-4359bd1bb004" />


📍 **Key findings:**  
- Cluster 0: Long-tenure, cashback users
- Cluster 1: High orders, low satisfaction
- Cluster 2: New, high app usage
- Cluster 3: Very new, low activity  

---


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
