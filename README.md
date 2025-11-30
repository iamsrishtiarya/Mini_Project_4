# Mini_Project_4


## 📱 **Decoding Mobile Usage Patterns in India**

This project explores how people across India use their mobile devices by applying both supervised and unsupervised machine learning techniques. By analyzing user behavior, device characteristics, and app/screen activity, the project uncovers usage patterns and segments users into meaningful behavioral groups.
All insights are showcased through a fully interactive **Streamlit dashboard** that supports real-time predictions and visual exploration.

---

## 🎯 **Objective**

The aim is to build a complete analytical ecosystem that can:

* Clean, preprocess, and organize raw device-level usage datasets
* Predict a user’s dominant device usage category through supervised learning
* Segment users into natural behavioral clusters using unsupervised models
* Present findings and predictions through an intuitive Streamlit interface

---

## 💼 **Business Applications**

| Use Case                   | Description                                                                       |
| -------------------------- | --------------------------------------------------------------------------------- |
| **Behavioral Profiling**   | Understand how different users engage with their devices across usage metrics     |
| **Device Improvements**    | Help OEMs optimize performance, UI design, and battery efficiency                 |
| **Personalized Offerings** | Assist telecom operators/app companies in customizing plans, alerts, and services |
| **Energy Optimization**    | Highlight battery usage trends and help users adopt power-saving habits           |

---

## 🧭 **Methodology**

### **1. Data Preparation**

* Dataset includes: user identifiers, OS information, phone model, app/screen time logs, and battery statistics
* Multiple raw files consolidated into a single structured dataset

### **2. Data Cleaning**

* Missing values treated using appropriate imputation strategies
* Standardized device model and operating system names
* Outliers handled using IQR and Z-score methods

### **3. Feature Engineering**

Engineered insight-rich features such as:

* **Total Usage Time**
* **Entertainment Index** (comprising social, gaming, and OTT usage)
* **Spend Ratio** (E-commerce expenditure vs recharge cost)
* **Age Group Binning**
* One-hot and ordinal encoding for categorical data
* Scaled numerical features using StandardScaler
* PCA used to reduce dimensionality for clustering and visualization

### **4. Exploratory Data Analysis**

* Trends in screen time, app category usage, and battery drain
* Correlation analysis with heatmaps and pairwise plots
* Distribution and behaviour of primary device usage classes

### **5. Machine Learning & Clustering**

#### 📌 **Classification Algorithms Used**

* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

**Model Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

#### 🧩 **Clustering Techniques**

* K-Means
* Hierarchical Clustering
* DBSCAN
* Gaussian Mixture Models
* Spectral Clustering

**Cluster Assessment:** Silhouette Score + PCA-based visualization
<img width="222" height="121" alt="image" src="https://github.com/user-attachments/assets/73839660-6fb5-44f6-b5ab-282e36e5b2fc" />

---

### 6. 🖥️ Application Development (Streamlit)
- Interactive EDA visualizations

  ![s1](https://github.com/user-attachments/assets/649277e9-52bf-46d4-92f4-c6102fc921ab)

  

- Input interface for real-time Primary Use prediction

  ![s2](https://github.com/user-attachments/assets/a84c65fb-9baa-4a6a-9572-aa51e0acc811)
- Clustering result visualization and user segmentation
  ![c1](https://github.com/user-attachments/assets/2f61aa3d-4044-4b80-ac6f-0c825c493da3)
  ![c2](https://github.com/user-attachments/assets/6cf568fb-92e5-49c8-b526-de8c78cb1c50)

### 7. ☁️ Deployment
- Deployed using Streamlit for web accessibility
---

## 🧰 Tech Stack

| Category        | Tools/Libraries |
|----------------|------------------|
| Programming     | Python       |
| Data Handling   | pandas, numpy    |
| Visualization   | seaborn, matplotlib, plotly |
| Machine Learning| scikit-learn, xgboost |
| Clustering      | scipy, sklearn   |
| Deployment      | Streamlit |

---

## 🧪 How to Run the Project Locally

1. **Clone the repository**
```bash
git clone https://github.com/SSaranya19/Decoding-Phone-Usage-Patterns-in-India.git
cd Decoding-Phone-Usage-Patterns-in-India


