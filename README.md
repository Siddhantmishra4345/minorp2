# 🧠 Human Activity Recognition (HAR) using Machine Learning

## 📌 Overview
This project focuses on Human Activity Recognition (HAR) using machine learning techniques.  
The goal is to classify human activities (such as walking, sitting, standing, etc.) based on smartphone sensor data.

The project uses the UCI HAR Dataset and implements a complete ML pipeline including preprocessing, training, and evaluation.

---

## 🎯 Objectives
- Build a machine learning model to classify human activities  
- Perform data preprocessing and feature handling  
- Train and evaluate models using real-world sensor data  
- Analyze model performance using metrics and visualization  

---

## 📊 Dataset
- Name: UCI Human Activity Recognition Dataset  
- Source: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones  

### Activities Classified:
- Walking  
- Walking Upstairs  
- Walking Downstairs  
- Sitting  
- Standing  
- Laying  

---

## ⚙️ Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## 🏗️ Project Structure

minorp2/ │ ├── har_trustworthy_pipeline_corrected_11.ipynb   # Main notebook ├── app.py                                       # Application script ├── pipeline.py                                  # ML pipeline ├── train_models.py                              # Model training ├── data_loader.py                               # Data loading ├── notebook_code.py                             # Helper notebook code ├── models/                                      # Saved models ├── utils/                                       # Utility functions ├── venv/                                        # Virtual environment (ignored) └── UCI-HAR Dataset-2/                           # Dataset (ignored)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
git clone https://github.com/Siddhantmishra4345/minorp2.git cd minorp2

### 2️⃣ Create virtual environment
python3 -m venv venv source venv/bin/activate

### 3️⃣ Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn xgboost notebook

### 4️⃣ Download Dataset
Download from:
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones  

Place it inside project folder:
minorp2/UCI-HAR Dataset/

---

### 5️⃣ Run Jupyter Notebook
jupyter notebook

Open:
har_trustworthy_pipeline_corrected_11.ipynb

---

## 📈 Model Workflow

1. Data Loading  
2. Data Preprocessing  
3. Feature Extraction  
4. Model Training (XGBoost)  
5. Model Evaluation  
6. Prediction  

---

## 📊 Results
- Model successfully classifies human activities  
- Achieved good accuracy on test data  
- Performance evaluated using confusion matrix and metrics  

---

## 🧠 Key Concepts Used
- Supervised Learning  
- Classification  
- Feature Engineering  
- Model Evaluation  
- Pipeline Design  

---

## ⚠️ Notes
- Dataset and virtual environment are excluded from GitHub using .gitignore  
- Ensure correct dataset path before running  

---

## 📌 Future Improvements
- Add deep learning models (CNN/LSTM)  
- Real-time activity prediction  
- Mobile app integration  

---

## 👨‍💻 Author
Siddhant Mishra  
GitHub: https://github.com/Siddhantmishra4345  

---

## ⭐ Acknowledgement
- UCI Machine Learning Repository  
- Open-source ML li
