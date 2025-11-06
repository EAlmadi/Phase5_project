# 🩺 Breast Cancer Survival and Treatment Prediction using Genomic and Histopathological Data  

## **1. Business Understanding**

### **1.1 Project Overview**
Breast cancer remains one of the leading causes of mortality among women worldwide. Advances in genomics and histopathology have enabled new opportunities to personalize treatment and improve survival outcomes.  

This project leverages **two complementary data sources**:  
-  **Genomic and clinical data** from the **METABRIC** dataset for survival and treatment prediction.  
-  **Histopathological image data** from the **BreakHis** dataset for tumor classification using deep learning.  

The project integrates **machine learning** (Logistic Regression, Random Forest, XGBoost) and **deep learning (CNN)** to build a system that:  
- Predicts **patient survival outcomes** and treatment recommendations.  
- Classifies **breast tissue images** as *benign* or *malignant*.  
- Deploys an interactive **web-based application** for clinicians and researchers.

---

### **1.2 Business Objectives**
1. Predict breast cancer **survival outcomes** using clinical and genomic data.  
2. Identify **key genomic and clinical factors** influencing treatment response.  
3. Classify **histopathological images** using CNN models.  
4. Recommend **treatment options** based on predictive modeling.  
5. Deploy the integrated system as an interactive **web app** for demonstration.

---

### **1.3 Research Questions**
1. Which clinical and genomic features best predict survival outcomes?  
2. Can we predict effective treatments (chemotherapy, radiotherapy, hormone therapy)?  
3. How accurately can CNNs classify benign and malignant breast tissue?  
4. Can integrating genomic and imaging data improve diagnostic accuracy?  

---

### **1.4 Success Criteria**
- **Model performance:**  
  - Genomic ML models: Accuracy ≥ 0.85, AUC ≥ 0.90  
  - CNN classifier: Accuracy ≥ 0.80  
- **Feature discovery:** Identify top 20 influential genomic or image-based features.  
- **Deployment:** Functional web app on **Render** (backend) and **Vercel** (frontend).  
- **Documentation:** Well-documented, reproducible GitHub project.  

---

## **2. Data Understanding**

### **2.1 Data Sources**
- **Clinical and Genomic Data:** [METABRIC — Breast Cancer Study (cBioPortal)](https://www.cbioportal.org/study/summary?id=brca_metabric)  
- **Histopathological Image Data:** [BreakHis Dataset (Kaggle)](https://www.kaggle.com/datasets/kurshidbasheer/breakhisbreast-cancer-histopathological-images)

---

### **2.2 Data Description**
- **METABRIC Dataset** (2,000 patients):  
  - Features: Age, Tumor Size, Stage, Grade, ER/PR/HER2 status, treatment types, mRNA gene expression levels (>20,000 genes).  
  - Target: Survival status and treatment type.  

- **BreakHis Dataset** (7,900 images):  
  - Classes: Benign vs Malignant.  
  - Magnifications: 40x, 100x, 200x, 400x.  
  - Used for CNN training.

---

### **2.3 Data Preparation**
Two datasets — clinical and mRNA  were merged on `Patient ID`, cleaned, and encoded for modeling.

```python
# Merging clinical and genomic data

clinical_df.rename(columns={'Patient ID': 'PATIENT_ID'}, inplace=True)
merged_df = clinical_df.merge(mrna_transposed, on='PATIENT_ID', how='inner')
```

## 3. CNN Model — Histopathological Image Classification

### 3.1 Overview  
A Convolutional Neural Network (CNN) was developed to classify breast tissue images from the **BreakHis dataset** into **Benign** and **Malignant** categories.  
This model supports the main project objective by detecting cancerous tissue from microscopic images, thereby assisting in early diagnosis and automated pathology analysis.

---

### 3.2 Dataset Description  
The dataset contains histopathological images of breast tumors captured under varying magnification levels.  

**Dataset Summary:**  
- **Patients:** Train – 56 | Validation – 13 | Test – 13  
- **Images:** Train – 5,388 | Validation – 1,247 | Test – 1,274  
- **Image Size:** 224 × 224 pixels  
- **Color Channels:** RGB  
- **Classes:** Benign (0) and Malignant (1)  

To address class imbalance, **class weights** were applied during model training.

---

### 3.3 Model Architecture  
The CNN model was constructed with multiple convolutional and pooling layers to extract hierarchical image features.  

**Architecture Summary:**  
| Layer Type | Parameters / Details |
|-------------|----------------------|
| Input | 224 × 224 × 3 (RGB) |
| Conv2D | 32 filters, kernel 3×3, ReLU activation |
| MaxPooling2D | 2×2 pool size |
| Conv2D | 64 filters, kernel 3×3, ReLU activation |
| MaxPooling2D | 2×2 pool size |
| Conv2D | 128 filters, kernel 3×3, ReLU activation |
| Dropout | 0.3 |
| Flatten | Converts 2D feature maps to 1D |
| Dense | 128 neurons, ReLU activation |
| Output | 1 neuron, Sigmoid activation (binary classification) |

---

### 3.4 Training Configuration  
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Batch Size:** 32  
- **Epochs:** 10  
- **Random Seed:** 42  
- **Validation Split:** 20%  

---

### 3.5 Model Performance  
| Metric | Training | Validation | Test |
|---------|-----------|-------------|------|
| **Accuracy** | 0.87 | 0.81 | 0.80 |
| **Loss** | 0.45 | 0.48 | 0.49 |

The results indicate stable generalization, with minimal overfitting between training and validation sets.

---

### 3.6 Model Evaluation  

**Training and Validation Loss Curve:**  
![Training and Validation Loss](f07e8cb2-f0e8-4612-a4e6-a19465b3b815.png)

**Confusion Matrix:**  
![Confusion Matrix](5f4a6254-135c-4602-ac46-d23f21309840.png)

**Sample Prediction (Benign):**  
![Prediction Example](5181dac4-307f-4780-aaf7-184edfaed072.png)

---

### 3.7 Insights  
1. The CNN achieved approximately **80% test accuracy**, effectively distinguishing benign from malignant tissues.  
2. The close alignment between training and validation accuracy suggests limited overfitting.  
3. Most misclassifications occurred in high-magnification images (400×), likely due to finer texture details.  
4. The model demonstrates potential for real-world diagnostic support when integrated with clinical and genomic data.

---

### 3.8 Recommendations  
1. Implement **Transfer Learning** using pre-trained models such as **VGG16**, **ResNet50**, or **InceptionV3** to enhance performance.  
2. Apply **data augmentation** (rotation, zoom, flipping) to improve robustness.  
3. Incorporate **Grad-CAM visualization** to interpret CNN focus areas and improve clinical interpretability.  
4. Combine CNN outputs with survival and treatment prediction models for **multi-modal cancer analysis**.  
5. Deploy the CNN within the final **web application** to enable image-based diagnostic predictions.

---
<h2 align="center">🚀 CNN App Deployment</h2>
<p align="center">
  <a href="https://neural-nexus-nine.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Open%20CNN%20App-Vercel-blue?style=for-the-badge&logo=vercel" alt="CNN App Link"/>
  </a>
</p>

## Authors
1. EAlmadi
2. 4068-tiffany
3. Hcton75
4. Khalidabdulkadir
5. Meshack7468