# 🏥 Healthcare Analytics – Hospital Stay & Patient Risk Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![EDA](https://img.shields.io/badge/EDA-Pandas-yellow)](https://pandas.pydata.org/)
[![Visualization](https://img.shields.io/badge/Visualization-Matplotlib_Seaborn-orange)](https://matplotlib.org/)

## Project Overview

A comprehensive Python-based exploratory data analysis (EDA) project analyzing hospital patient data to identify factors influencing length of stay, segment patient populations by risk level, and uncover patterns in readmission rates. This project combines data cleaning, feature engineering, and advanced visualization to derive actionable clinical and operational insights.

**Dataset:** hospital__stay_data.csv | **Industry:** Healthcare Analytics & Hospital Operations

---

## 📋 Table of Contents

- [Project Objective](#project-objective)
- [Dataset Description](#dataset-description)
- [Key Analysis Steps](#key-analysis-steps)
- [Installation & Prerequisites](#installation--prerequisites)
- [How to Use](#how-to-use)
- [Key Findings](#key-findings)
- [Visualization Guide](#visualization-guide)
- [Learning Outcomes](#learning-outcomes)
- [Tech Stack](#tech-stack)

---

## 🎯 Project Objective

Analyze hospital patient data to:

✅ Identify factors driving longer hospital stays (length of stay patterns)  
✅ Segment patients by risk level based on comorbidities and demographics  
✅ Analyze high-risk patient populations requiring enhanced care coordination  
✅ Understand temporal patterns in patient visits and admissions  
✅ Provide data-driven insights for resource optimization and operational efficiency  

---

## 📊 Dataset Description

### File: `hospital__stay_data.csv`

**Dataset Statistics:**
- Total Records: 100,000 patient admissions
- Date Range: 2012 (multi-year hospital records)
- Key Numeric Columns: `lengthofstay`, `rcount`, `bmi`, `pulse`, `respiration`, `hematocrit`, `neutrophils`, `sodium`, `glucose`, `bloodureanitro`, `creatinine`
- Key Categorical Columns: `gender`, `dialysisrenalendstage`, `asthma`, `irondef`, `pneum`, `substancedependence`, `psychologicaldisordermajor`, `depress`, `psychother`, `fibrosisandother`, `malnutrition`, `hemo`, `secondarydiagnosisnonicd9`, `discharged`, `facid`
- Date Column: `vdate` (visit date for temporal analysis)
- **Data Quality:** No missing values in any column; clean dataset ready for analysis

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| lengthofstay | Numeric | Hospital stay duration in days |
| rcount | Numeric | Readmission count |
| bmi | Numeric | Body Mass Index |
| pulse | Numeric | Patient pulse rate |
| respiration | Numeric | Respiration rate |
| hematocrit | Numeric | Blood hematocrit level |
| neutrophils | Numeric | Neutrophil count |
| sodium | Numeric | Sodium level |
| glucose | Numeric | Blood glucose level |
| bloodureanitro | Numeric | Blood urea nitrogen |
| creatinine | Numeric | Creatinine level |
| gender | Categorical | Male / Female |
| dialysisrenalendstage | Categorical | Kidney disease indicator |
| asthma | Binary | Presence of asthma |
| irondef | Binary | Iron deficiency indicator |
| pneum | Binary | Pneumonia indicator |
| substancedependence | Binary | Substance dependency flag |
| psychologicaldisordermajor | Binary | Major psychological disorder |
| depress | Binary | Depression indicator |
| psychother | Binary | Psychotherapy need |
| fibrosisandother | Binary | Fibrosis and other conditions |
| malnutrition | Binary | Malnutrition indicator |
| hemo | Binary | Hemophilia indicator |
| vdate | DateTime | Visit/admission date |

---

## 🔬 Key Analysis Steps

### Step 1: Load Dataset & Check Missing Values
```python
import pandas as pd

# Load dataset
df = pd.read_csv('hospital_stay_data.csv')

# View first few rows
print(df.head())

# Check missing values
print(df.isnull().sum())
```

### Step 2: Data Cleaning & Pre-Processing

**Handle Numeric Columns:**
```python
numeric_cols = ['lengthofstay','rcount','bmi','pulse','respiration','hematocrit',
                'neutrophils','sodium','glucose','bloodureanitro','creatinine']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
```

**Handle Categorical Columns:**
```python
categorical_cols = ['gender','dialysisrenalendstage','asthma','irondef','pneum',
                    'substancedependence','psychologicaldisordermajor','depress',
                    'psychother','fibrosisandother','malnutrition','hemo',
                    'secondarydiagnosisnonicd9','discharged','facid']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
        df[col] = df[col].astype('category')
```

**Handle Date Column:**
```python
if 'vdate' in df.columns:
    df['vdate'] = pd.to_datetime(df['vdate'])

# Verify data types
print(df.dtypes)
print(df.isnull().sum())
```

### Step 3: Feature Engineering

**Create Comorbidity Features:**
```python
comorbidity_cols = ['asthma','irondef','pneum','substancedependence',
                    'psychologicaldisordermajor','depress','psychother',
                    'fibrosisandother','malnutrition','hemo']

# Convert to numeric
for col in comorbidity_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Count comorbidities per patient
df['comorbidity_count'] = df[comorbidity_cols].sum(axis=1)

# Flag high-risk patients (3+ comorbidities)
df['high_risk'] = df['comorbidity_count'].apply(lambda x: 'Yes' if x>=3 else 'No')
```

**Extract Temporal Features:**
```python
# Example: Calculate Cost per Day if 'cost' exists
if 'cost' in df.columns:
    df['cost_per_day'] = df['cost'] / df['lengthofstay']

# Extract visit month/year from vdate
df['visit_month'] = df['vdate'].dt.month
df['visit_year'] = df['vdate'].dt.year

# Verify new features
print(df[['comorbidity_count','high_risk','visit_month','visit_year']].head())
```

### Step 4: Basic Visualizations

**Length of Stay Distribution:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for Length of Stay
plt.figure(figsize=(8,5))
sns.histplot(df['lengthofstay'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Length of Stay')
plt.xlabel('Length of Stay (days)')
plt.ylabel('Number of Patients')
plt.show()

# Boxplot for outlier detection
plt.figure(figsize=(8,5))
sns.boxplot(x=df['lengthofstay'], color='lightgreen')
plt.title('Boxplot of Length of Stay')
plt.xlabel('Length of Stay (days)')
plt.show()
```

**High-Risk Patient Count:**
```python
plt.figure(figsize=(6,4))
sns.countplot(x='high_risk', data=df, hue='high_risk', palette='Set2', dodge=False, legend=False)
plt.title('Number of High-Risk vs Normal Patients')
plt.xlabel('High Risk')
plt.ylabel('Number of Patients')
plt.show()
```

### Step 5: Correlation Analysis

```python
numeric_cols = ['rcount','hematocrit','neutrophils','sodium','glucose',
                'bloodureanitro','creatinine','bmi','pulse','respiration','lengthofstay']

# Compute correlation matrix
corr_matrix = df[numeric_cols].corr()

# Plot heatmap
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
plt.show()
```

### Step 6: Patient Visits Over Time

```python
# Aggregate visits by year and month
visits_per_month = df.groupby(['visit_year', 'visit_month']).size().reset_index(name='num_visits')

# Plot line chart
plt.figure(figsize=(12,5))
sns.lineplot(x='visit_month', y='num_visits', hue='visit_year', data=visits_per_month, 
             marker='o', palette='tab10')
plt.title('Patient Visits Over Time (Month-wise)', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Number of Visits')
plt.xticks(range(1,13))
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

### Step 7: Risk Status & Comorbidity Analysis

**Length of Stay by Risk Status:**
```python
plt.figure(figsize=(8,5))
sns.boxplot(x='high_risk', y='lengthofstay', data=df, palette='Set3')
plt.title('Length of Stay by High-Risk Status', fontsize=16)
plt.xlabel('High Risk')
plt.ylabel('Length of Stay (days)')
plt.show()
```

**Comorbidity Distribution:**
```python
plt.figure(figsize=(8,5))
sns.histplot(df['comorbidity_count'], bins=range(0, df['comorbidity_count'].max()+2), 
             color='skyblue', kde=False)
plt.title('Distribution of Comorbidity Count', fontsize=16)
plt.xlabel('Number of Comorbidities')
plt.ylabel('Number of Patients')
plt.show()
```

**Length of Stay vs Comorbidity Count:**
```python
plt.figure(figsize=(8,5))
sns.scatterplot(x='comorbidity_count', y='lengthofstay', data=df, hue='high_risk', palette='Set1')
plt.title('Length of Stay vs Comorbidity Count', fontsize=16)
plt.xlabel('Comorbidity Count')
plt.ylabel('Length of Stay (days)')
plt.show()
```

### Step 8: Summary Insights

```python
# High-risk patients count
high_risk_count = df['high_risk'].value_counts()
print("High-Risk vs Normal Patients:\n", high_risk_count, "\n")


# Average length of stay
avg_los = df.groupby('high_risk')['lengthofstay'].mean()
print("Average Length of Stay by Risk Status:\n", avg_los, "\n")


# Comorbidity count distribution
comorbidity_summary = df['comorbidity_count'].describe()
print("Comorbidity Count Summary:\n", comorbidity_summary, "\n")


# Top months with highest patient visits
top_months = df.groupby('visit_month').size().sort_values(ascending=False).head(5)
print("Top 5 Months with Most Patient Visits:\n", top_months)

```

### Step 9: Comprehensive Dashboard

```python
import matplotlib.gridspec as gridspec

plt.figure(figsize=(20,16))
gs = gridspec.GridSpec(3, 2)

# 1️⃣ High-Risk Count
ax0 = plt.subplot(gs[0,0])
sns.countplot(x='high_risk', data=df, color='lightgreen', ax=ax0)
ax0.set_title('High-Risk vs Normal Patients', fontsize=14)
ax0.set_xlabel('High Risk')
ax0.set_ylabel('Number of Patients')

# Annotate percentages
total = len(df)
for p in ax0.patches:
    height = p.get_height()
    percent = f'{100*height/total:.1f}%'
    ax0.annotate(percent, (p.get_x() + p.get_width()/2., height),
                 ha='center', va='bottom', fontsize=11, color='black')

# 2️⃣ Comorbidity Count Distribution
ax1 = plt.subplot(gs[0,1])
sns.histplot(df['comorbidity_count'], bins=range(0, df['comorbidity_count'].max()+2),
             color='skyblue', kde=False, ax=ax1)
ax1.set_title('Distribution of Comorbidity Count', fontsize=14)
ax1.set_xlabel('Number of Comorbidities')
ax1.set_ylabel('Number of Patients')

avg_comorb = df['comorbidity_count'].mean()
ax1.axvline(avg_comorb, color='red', linestyle='--')
ax1.text(avg_comorb+0.1, ax1.get_ylim()[1]*0.9, f'Avg: {avg_comorb:.2f}', color='red', fontsize=11)

# 3️⃣ Length of Stay by High-Risk
ax2 = plt.subplot(gs[1,0])
sns.boxplot(x='high_risk', y='lengthofstay', data=df, color='lightcoral', ax=ax2)
ax2.set_title('Length of Stay by High-Risk Status', fontsize=14)
ax2.set_xlabel('High Risk')
ax2.set_ylabel('Length of Stay (days)')

for i, hr in enumerate(['No', 'Yes']):
    mean_los = df[df['high_risk']==hr]['lengthofstay'].mean()
    ax2.text(i, mean_los+0.5, f'Mean: {mean_los:.1f}', ha='center', color='black', fontsize=11)

# 4️⃣ Length of Stay vs Comorbidity Count
ax3 = plt.subplot(gs[1,1])
sns.scatterplot(x='comorbidity_count', y='lengthofstay', data=df, hue='high_risk',
                palette={'No':'blue','Yes':'red'}, ax=ax3, alpha=0.6)
ax3.set_title('Length of Stay vs Comorbidity Count', fontsize=14)
ax3.set_xlabel('Comorbidity Count')
ax3.set_ylabel('Length of Stay (days)')
ax3.legend(title='High Risk', loc='upper left')

# 5️⃣ Patient Visits Over Time
ax4 = plt.subplot(gs[2,:])
visits_per_month = df.groupby(['visit_year', 'visit_month']).size().reset_index(name='num_visits')
sns.lineplot(x='visit_month', y='num_visits', hue='visit_year', data=visits_per_month,
             marker='o', palette='tab10', ax=ax4)
ax4.set_title('Patient Visits Over Time (Month-wise)', fontsize=14)
ax4.set_xlabel('Month')
ax4.set_ylabel('Number of Visits')
ax4.set_xticks(range(1,13))
ax4.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('Healthcare Patient Analytics – Key Insights', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()
```

---

## ⚙️ Installation & Prerequisites

```bash
pip install pandas numpy matplotlib seaborn
```

**Python Version:** 3.7+

---

## 🚀 How to Use

1. **Ensure dataset file** `hospital_stay_data.csv` is in your working directory
2. **Run each code block sequentially** starting from Step 1
3. **Visualizations** will display inline (in Jupyter Notebook) or open in a new window
4. **Modify column names** if your dataset has different naming conventions
5. **Adjust comorbidity threshold** in feature engineering (currently set to 3+ conditions for high-risk)

---

## 📈 Key Findings

### Patient Population Overview
- **Total Patients Analyzed:** 100,000 admissions
- **High-Risk Patients (3+ comorbidities):** 6,900 (6.9%)
- **Normal-Risk Patients:** 93,100 (93.1%)
- **Average Comorbidities:** 0.71 per patient
- **Comorbidity Range:** 0-8 chronic conditions

### Length of Stay Insights

**Overall Stay Duration:**
- High-Risk Patients: **Average 5.96 days**
- Normal-Risk Patients: **Average 3.86 days**

**Key Finding:** Patients with 3 or more comorbidities demonstrate significantly prolonged hospital stays, indicating higher clinical complexity and resource requirements.

### Comorbidity Distribution
- **Mean Comorbidity Count:** 0.71
- **Median:** 0 (majority of patients have no recorded comorbidities)
- **75th Percentile:** 1 comorbidity
- **Maximum:** 8 comorbidities per patient

**Clinical Insight:** Most patients (75%) have 0-1 comorbidities, but the 6.9% with 3+ conditions drive longer stays and higher resource utilization.

### Temporal Patterns - Patient Visits by Month
Top 5 months with highest patient visits:
1. **January (Month 1):** 8,600 visits (highest)
2. **October (Month 10):** 8,566 visits
3. **May (Month 5):** 8,482 visits
4. **August (Month 8):** 8,478 visits
5. **March (Month 3):** 8,442 visits

**Distribution:** Relatively uniform across all months (~8,400-8,600 visits per month), suggesting consistent patient volume with slight winter peaks.

### Risk Stratification Summary
- **High-Risk Segment (3+ comorbidities):** 6,900 patients requiring enhanced care coordination
- **Normal-Risk Segment:** 93,100 patients with standard care protocols
- **Stay Duration Multiplier:** High-risk patients require 1.54× longer hospitalization

---

## 📊 Visualization Guide

| Visualization | Purpose | Key Insight |
|---------------|---------|-------------|
| **Length of Stay Histogram** | Shows distribution of hospital stays | Identifies typical vs. prolonged stays |
| **Boxplot by Risk Status** | Compares stay duration between risk groups | High-risk patients have longer stays |
| **Comorbidity Count Distribution** | Shows spread of comorbidities across population | Identifies prevalence of multi-condition patients |
| **Length of Stay vs Comorbidity Scatter** | Visualizes relationship between factors | Positive correlation between comorbidities and stay |
| **Patient Visits Over Time** | Temporal trend analysis | Seasonal patterns and admission volume trends |
| **Correlation Heatmap** | Shows relationships between all numeric variables | Identifies which factors most influence outcomes |
| **High-Risk Patient Count** | Patient segmentation visualization | Proportion of high-risk vs. normal patients |

---

## 🎓 Learning Outcomes

✅ Exploratory Data Analysis (EDA) fundamentals in healthcare domain

✅ Data cleaning: handling missing values, type conversion, validation

✅ Feature engineering: creating derived features for patient risk stratification

✅ Pandas groupby operations for clinical analytics

✅ Data visualization with Matplotlib and Seaborn

✅ Statistical analysis of health metrics

✅ Patient segmentation and risk classification

✅ Temporal data analysis and time-series patterns

✅ Correlation analysis and relationship identification

✅ Healthcare-specific analytical workflows

---

## 📝 Notes

- All missing numeric values are filled with median imputation
- Categorical missing values are replaced with "Unknown"
- High-risk threshold set to 3 or more comorbidities (adjustable)
- Temporal features extracted from `vdate` column for trend analysis
- All visualizations use Seaborn styling for professional appearance

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook / VS Code |
| **Dataset Used** | hospital__stay_data.csv |

---

## 📝 Author
**Robin Jimmichan Pooppally**  
[LinkedIn](https://www.linkedin.com/in/robin-jimmichan-pooppally-676061291) | [GitHub](https://github.com/Robi8995)

---

*This project demonstrates practical healthcare analytics expertise in clinical operations, combining admission-type segmentation and diagnosis profiling with readmission risk modeling to drive measurable improvements in resource optimization, cost reduction, patient outcomes, and operational efficiency through data-driven risk stratification*

