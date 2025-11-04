# Breast Cancer Survival and Treatment Prediction using Genomic and Histopathological Data

## Business Understanding

### 1.1 Project Overview
Breast cancer remains one of the leading causes of mortality among women worldwide. Advances in genomics and clinical data collection have created opportunities to better understand cancer behavior and guide personalized treatment strategies.  

The **METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)** dataset provides detailed **clinical**, **pathological**, and **gene-expression** information for nearly 2,000 breast cancer patients. This makes it ideal for developing data-driven systems that can help predict patient outcomes and recommend potential treatment approaches. 

This project aims to combine **machine learning**, **statistical analysis**, and **data-driven modeling** to build a predictive system that:
- Estimates **patient survival outcomes**, and  
- Suggests **potential treatment options** (such as chemotherapy, hormone therapy, or targeted therapy) based on similar patient profiles.  

It combines **traditional ML models** (Logistic Regression, Random Forest, XGBoost) with a **Convolutional Neural Network (CNN)** to analyze both structured and image data — offering a holistic, data-driven view of cancer prognosis and therapy optimization.

Ultimately, the system will be deployed as a **web-based application (Django + React)** that clinicians and researchers can use to explore predictions and recommendations interactively.

---

### 1.2 Business Objectives
The primary objectives of this project are:
1. **Predict survival outcomes** using both clinical and gene-expression data.  
2. **Identify key clinical and genomic factors** that influence patient survival and treatment response.  
3. **Recommend possible treatment paths** based on patterns learned from patients with similar profiles and treatment histories.  
4. **Deploy** the model as a web application for demonstration and educational purposes.

---

### 1.3 Research Questions
This project will seek to answer the following key research questions:

1. **Predictive Insight:**  
   Which clinical and genetic features are most predictive of breast cancer survival outcomes?  

2. **Treatment Recommendation:**  
   Can we identify the most effective treatment approaches (e.g., hormone therapy, radiotherapy, chemotherapy) for patients with specific gene-expression profiles?  

3. **Pattern Discovery:**  
   How do different molecular subtypes (e.g., Luminal A, Basal-like, HER2-enriched) correlate with overall survival and treatment response?  

4. **Model Interpretability:**  
   How can the model’s predictions and recommendations be made interpretable and clinically meaningful for decision support?

---

### 1.4 Project Success Criteria
The success of the project will be measured through:
- **Model performance:**  
  - Survival prediction models: Accuracy ≥ 0.85, AUC ≥ 0.90  
  - Treatment recommendation models: Top-3 accuracy ≥ 0.75  
- **Feature discovery:** Identification of the top 20 features (genes + clinical attributes) that most influence survival and treatment outcomes.  
- **Deployment:** Fully functional web application (Django backend, React frontend) hosted on **Render** and **Vercel**.  
- **Collaboration & Documentation:** Reproducible workflow managed on GitHub and documented in Notion.

---

> **Summary:**  
> This phase establishes the business motivation, project scope, objectives, and research focus.  
> The next phase — *Data Understanding* — will involve exploring the METABRIC dataset in detail, performing EDA, and identifying relationships between features, survival outcomes, and treatment variables.


## 2. Data Understanding  

### 2.1 Data Collection  
The data used in this project was obtained from the **[cBioPortal for Cancer Genomics](https://www.cbioportal.org/)**, specifically from the **METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)** study.  
The METABRIC dataset is a large-scale breast cancer resource that integrates both **clinical** and **genomic** information from nearly **2,000 patients**, making it one of the most comprehensive datasets available for breast cancer research.  

Two datasets were downloaded and later merged:  
- **Clinical Data:** Containing patient demographics, tumor characteristics, receptor status, treatment details, and survival outcomes.  
- **mRNA Expression Data:** Containing expression levels for over **20,000 genes**, capturing molecular-level tumor variations.  

**Dataset link:**  
🔗 [METABRIC (Breast Cancer) — cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)

 **Image Dataset (for CNN)**  
   - [BreakHis Breast Cancer Histopathological Images (Kaggle)](https://www.kaggle.com/datasets/kurshidbasheer/breakhisbreast-cancer-histopathological-images)  
   - Microscopic images used to train a CNN for tumor classification.


---

### 2.2 Data Description  
- **Clinical Dataset:** Includes variables such as  
  *Age at Diagnosis, Tumor Size, Tumor Stage, Histologic Grade, ER/PR/HER2 Status, Menopausal State, Treatment Type (Chemotherapy, Hormone Therapy, Radiotherapy), and Survival Information (Overall Survival Status and Duration).*  

- **mRNA Dataset:** Includes high-dimensional gene expression values (over 20,000 gene features) for each patient. These features represent the expression intensity of various genes and serve as genomic indicators of tumor behavior.  

The two datasets were **merged using the Patient ID** as the key identifier, resulting in a single integrated dataset containing both **clinical** and **molecular** attributes.

---

### 2.3 Data Quality and Preparation Notes  
Initial inspection showed that both datasets were well-structured but contained missing values in some clinical attributes (e.g., receptor status, treatment indicators).  
Appropriate cleaning techniques were applied, including:  
- Handling missing or inconsistent entries  
- Encoding categorical features  
- Converting data types for modeling  

Due to the **curse of dimensionality** associated with over 20,000 genomic features, feature selection was performed.  
Instead of using all genes, we focused on a subset of **biologically significant genes** (e.g., *BRCA1, BRCA2, TP53, ERBB2*) based on **domain knowledge and breast cancer literature**.  
This approach improved interpretability and model performance.

---

### 2.4 Initial Data Exploration  
Exploratory analysis was conducted to understand the distribution of clinical variables and survival outcomes.  
Key patterns observed include:  
- Most patients were diagnosed between the ages of 45–70.  
- Early-stage tumors (Stage 1) were associated with higher survival rates, while advanced stages (Stages 3–4) showed higher mortality.  
- Hormone receptor status (ER, PR, HER2) showed strong relationships with treatment types and outcomes.  

These insights guided feature selection and informed the modeling strategy used in subsequent stages.

---

### 📚 Data Citation  
Cerami, E., Gao, J., Dogrusoz, U. *et al.* (2012). **The cBio Cancer Genomics Portal: An Open Platform for Exploring Multidimensional Cancer Genomics Data.** *Cancer Discovery, 2(5), 401–404.*  
Accessed from [https://www.cbioportal.org/study/summary?id=brca_metabric](https://www.cbioportal.org/study/summary?id=brca_metabric)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

```
___

# 3. Data Preparation

## 3.1 Data Merging
Two datasets — **clinical** and **mRNA expression**  were merged using the common identifier **Patient ID**.  
This integration created a single dataset that combines both clinical characteristics and genomic features, enabling a comprehensive analysis of breast cancer outcomes and treatment responses.

---

```python
#merging
#renaming patient_id in the clinical_df to match PATIENT_ID in the mrna_transposed
clinical_df.rename(columns={'Patient ID': 'PATIENT_ID'}, inplace=True)
#merging on PATIENT_ID
merged_df = clinical_df.merge(mrna_transposed, on='PATIENT_ID', how='inner')
merged_df.head()
```
    

## Key Insights and Recommendations

### Survival Prediction Insights

Three models, Logistic Regression, SVM, and XGBoost were trained to predict overall survival. Among them, XGBoost achieved the best performance with an accuracy of approximately 79%, followed by Logistic Regression at 77% and SVM at 76%.

These results indicate that combining gene expression profiles with clinical variables provides a reliable basis for estimating patient survival. Logistic Regression and SVM offered consistent but slightly lower performance, suggesting linear relationships capture part of the signal, while XGBoost better models complex, non-linear interactions.

The correlation analysis revealed that certain genes, such as IL6ST, TYW3, and KIF13B, were positively correlated with overall survival, while Age at Diagnosis showed a mild negative correlation, aligning with clinical expectations that younger patients often have better outcomes.

Treatment Prediction Insights

Models were also developed to predict whether patients received Chemotherapy, Radio Therapy, or Hormone Therapy. Across all algorithms, XGBoost consistently outperformed others, reaching about 87% accuracy for Chemotherapy, 77% for Hormone Therapy, and around 61% for Radio Therapy.

Chemotherapy and hormone therapy predictions showed strong alignment between the model and actual treatment outcomes, suggesting these are influenced by measurable genomic and clinical factors. However, radio therapy predictions were weaker, likely due to additional clinical decision factors not captured in the dataset — such as tumor location or physician preference.

### Key Insights

1. XGBoost is the most effective model across both survival and treatment prediction tasks, demonstrating its strength in handling complex relationships between genetic and clinical data.

2. Chemotherapy and hormone therapy outcomes can be predicted with high confidence, showing clear patterns in the data.

3. Radio therapy predictions remain modest, indicating missing predictors or higher variability in clinical decision-making.

4. Certain genes, particularly IL6ST, TYW3, and KIF13B may serve as potential biomarkers associated with improved survival.

5. Age at diagnosis negatively impacts survival, confirming known clinical patterns.

### Recommendations

1. Adopt XGBoost as the primary predictive model for both survival and treatment prediction, given its superior performance and reliability.

2. Incorporate additional clinical features such as tumor grade, histological type, and receptor status (ER, PR, HER2) to improve model performance, especially for radio therapy prediction.

3. Analyze feature importance from XGBoost models to identify key genes driving survival and treatment outcomes, potentially informing biomarker discovery and personalized treatment strategies.

4. Validate models using cross-validation and external datasets to ensure generalizability and reduce overfitting risks.

5. Develop a user-friendly interface (e.g., Streamlit or Flask) to enable clinicians to input patient data and receive predictive insights on expected survival and suitable treatments.

6. Continue integrating genomic data with clinical decision support, as the findings demonstrate that machine learning models can meaningfully assist in predicting patient outcomes and optimizing treatment strategies.


<h2 align="center">🚀 Live Demo</h2>
<p align="center">
  <a href="https://neural-nexus-nine.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Open%20App-Vercel-blue?style=for-the-badge&logo=vercel" alt="App Link"/>
  </a>
</p>

---

### Authors
Elsie
Tiffany
Khalid
Jeremy
Meshack
