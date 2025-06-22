
# 🏥 Medical Insurance Cost Prediction

This project aims to build a machine learning model that predicts individual medical insurance costs based on personal and health-related features. The dataset contains features such as age, gender, BMI, smoking status, and region.

## 📁 Project Structure

```
Medical_Insurance_Cost_Prediction/
│
├── Medical_insurance.ipynb   # Main Jupyter Notebook with complete analysis
├── README.md                 # Project documentation (this file)
└── models/                   # (Optional) Folder for saving trained models
```

## 📊 Dataset Description

The dataset contains the following key features:

- **age**: Age of the individual
- **sex**: Gender of the individual
- **bmi**: Body Mass Index
- **children**: Number of children covered by health insurance
- **smoker**: Smoking status (`yes` or `no`)
- **region**: Residential region in the US
- **charges**: Medical insurance cost (target variable)

## ⚙️ Technologies Used

- Python 🐍
- NumPy, Pandas (for data handling)
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (for ML modeling)
- XGBoost, SVR, KNN, RandomForestRegressor (for regression tasks)

## 🧪 Project Workflow

1. **Data Preprocessing**
   - Missing value handling
   - Outlier detection and treatment (Z-score & IQR methods)
   - Categorical encoding
   - Feature scaling
   - Feature engineering (polynomial & log transformations)

2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis
   - Correlation heatmaps
   - Feature importance plots

3. **Model Building**
   - Models: Random Forest, XGBoost, SVR, KNN
   - Evaluation Metrics: RMSE, MAE, R² Score

4. **Best Model**
   - **Random Forest Regressor** yielded the best performance with **R² ≈ 0.9266**

5. **Deployment-ready**
   - Final model can be saved using `joblib` or `pickle` for deployment.

## 📈 Performance Metrics

| Model               | R² Score | RMSE    | MAE     |
|--------------------|----------|---------|---------|
| Random Forest       | 0.9266   | ~2100   | ~1500   |
| XGBoost             | ...      | ...     | ...     |
| SVR                 | ...      | ...     | ...     |
| KNN                 | ...      | ...     | ...     |

(*Replace `...` with actual values from your notebook.*)

## 💡 Key Insights

- **Smoker status**, **BMI**, and **Age** have the most significant impact on charges.
- Handling outliers and applying transformations improves model performance considerably.

## 🚀 Future Enhancements

- Hyperparameter tuning (GridSearchCV)
- Model ensembling
- Streamlit-based web deployment
- API endpoint creation using Flask/FastAPI

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-insurance-cost-prediction.git
   cd medical-insurance-cost-prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Medical_insurance.ipynb
   ```
