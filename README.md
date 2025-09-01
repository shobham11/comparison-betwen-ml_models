# Machine Learning Model Comparison

## 📌 Project Overview

This project compares the performance of various machine learning models on the **Campaign Responses Dataset** (`campaign_responses.csv`).
The goal is to predict whether a customer will respond positively (`Yes/No`) to a campaign, based on features such as income, family size, and credit score.

## 📂 Dataset

* **File:** `campaign_responses.csv`
* **Target Variable:** `responded` (mapped as `Yes=1`, `No=0`)
* **Features:** Annual income, family size, credit score, and derived ratios

## ⚙️ Workflow

1. **Data Preprocessing**

   * Handling categorical and numerical variables
   * Feature engineering (e.g., income-to-family-size ratio)
   * Converting labels (`Yes/No` → `1/0`)

2. **Exploratory Data Analysis (EDA)**

   * Summary statistics
   * Correlation heatmap
   * Scatter plots to visualize relationships

3. **Model Training & Comparison**
   Implemented and compared multiple models:

   * Logistic Regression
   * Decision Tree
   * Random Forest
   * Support Vector Machine (SVM)
   * K-Nearest Neighbors (KNN)
   * Gradient Boosting / XGBoost (if available)

4. **Evaluation Metrics**

   * Accuracy
   * Precision, Recall, F1-score
   * Confusion Matrix

## 📊 Results

* Models were compared to identify the best performer on this classification task.
* Detailed results (metrics and visualizations) are included in the notebook.

## 🛠️ Requirements

Make sure the following libraries are installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 🚀 Usage

1. Open the Jupyter Notebook:

   ```bash
   jupyter notebook ML-Models.ipynb
   ```
2. Run the cells step by step to:

   * Load and preprocess data
   * Explore the dataset visually
   * Train multiple models
   * Compare results

## 📌 Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Feature selection methods for dimensionality reduction
* Cross-validation for more robust performance estimates

