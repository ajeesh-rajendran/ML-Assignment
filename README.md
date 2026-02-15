# Car Price Classification — ML Assignment 2

## a. Problem Statement

The objective of this project is to implement and compare multiple classification models on a car price dataset. Given various features of a car (brand, body type, fuel type, engine specifications, etc.), the goal is to classify each car into a price category (**Budget**, **Mid-Range**, or **Premium**). Six different ML classification algorithms are trained, evaluated, and deployed via an interactive Streamlit web application.

## b. Dataset Description

- **Dataset**: `global_cars_enhanced.csv`
- **Source**: Public car dataset with enhanced features
- **Instances**: 300
- **Features**: 16 columns

| Column | Description |
|---|---|
| Car_ID | Unique identifier for each car |
| Brand | Car manufacturer (Mercedes, Nissan, etc.) |
| Manufacture_Year | Year the car was manufactured |
| Body_Type | Type of car body (SUV, Coupe, Hatchback, Sedan, etc.) |
| Fuel_Type | Fuel type (Petrol, Diesel, etc.) |
| Transmission | Transmission type (Manual, Automatic) |
| Engine_CC | Engine displacement in CC |
| Horsepower | Engine horsepower |
| Mileage_km_per_l | Fuel efficiency in km/l |
| Price_USD | Car price in USD |
| Manufacturing_Country | Country of manufacture |
| Car_Age | Age of the car in years |
| Price_Category | **Target variable** — Budget, Mid-Range, Premium |
| HP_per_CC | Horsepower per CC ratio |
| Age_Category | Categorical age group (New, Old) |
| Efficiency_Score | Computed efficiency score |

- **Target Variable**: `Price_Category` (Budget, Mid-Range, Premium)
- **Preprocessing**: Dropped `Car_ID` (non-informative) and `Price_USD` (target leakage). All categorical features were label-encoded. Features were scaled using `StandardScaler`.

## c. Models Used

Six classification models were implemented and evaluated on the same dataset using an 80/20 train-test split with stratification.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.3333 | 0.5258 | 0.3356 | 0.3333 | 0.3299 | 0.1140 |
| Decision Tree | 0.3000 | 0.5786 | 0.3084 | 0.3000 | 0.2873 | 0.0660 |
| kNN | 0.2333 | 0.4772 | 0.2433 | 0.2333 | 0.2302 | -0.0226 |
| Naive Bayes | 0.3167 | 0.5328 | 0.3164 | 0.3167 | 0.3073 | 0.0943 |
| Random Forest (Ensemble) | 0.2667 | 0.5424 | 0.2765 | 0.2667 | 0.2660 | 0.0263 |
| XGBoost (Ensemble) | 0.3333 | 0.5627 | 0.3495 | 0.3333 | 0.3277 | 0.1116 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves the joint-highest accuracy (33.33%) along with XGBoost. Provides a good baseline with stable performance across precision, recall, and F1. Suitable for linearly separable class boundaries. |
| Decision Tree | Accuracy is 30.00%, but achieves the highest AUC (0.5786) among all models, suggesting better class probability separation. Prone to overfitting on smaller datasets despite regularization via max_depth and min_samples constraints. |
| kNN | Lowest performing model with 23.33% accuracy and negative MCC (-0.0226), indicating predictions are worse than random for this dataset. Feature scaling helps but the small dataset size and overlapping class distributions limit kNN's effectiveness. |
| Naive Bayes | Achieves 31.67% accuracy. The Gaussian assumption works reasonably well given the mixed feature space. Performs comparably to Logistic Regression despite the strong feature independence assumption. |
| Random Forest (Ensemble) | Accuracy of 26.67% is below expectations for an ensemble method. The limited dataset size (300 rows) may not provide enough diversity for the ensemble of 100 trees to generalize well. AUC of 0.5424 is moderate. |
| XGBoost (Ensemble) | Ties with Logistic Regression for the highest accuracy (33.33%) and achieves the best precision (0.3495) among all models. Gradient boosting's sequential error correction provides marginal improvement. Second-highest AUC (0.5627). |

## Project Structure

```
ML Assignment/
│── app.py                         # Streamlit web application
│── requirements.txt               # Python dependencies
│── README.md                      # This file
│── Car_Price_Classification.ipynb  # Training notebook
│── global_cars_enhanced.csv       # Dataset
│── Models/                        # Saved model files (.joblib)
│   ├── Logistic_Regression.joblib
│   ├── Decision_Tree.joblib
│   ├── KNN.joblib
│   ├── Gaussian_Naive_Bayes.joblib
│   ├── Random_Forest.joblib
│   ├── XGBoost.joblib
│   ├── scaler.joblib
│   ├── label_encoders.joblib
│   ├── target_encoder.joblib
│   └── feature_names.joblib
└── test_data/                     # Sample test CSV files
    ├── test_data_1.csv
    ├── test_data_2.csv
    ├── test_data_3.csv
    ├── test_data_4.csv
    └── test_data_5.csv
```

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train models** (if not already saved):
   Open and run all cells in `Car_Price_Classification.ipynb`

3. **Launch Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Use the app**:
   - Select a model from the dropdown
   - Upload a test CSV file
   - View evaluation metrics, classification report, and confusion matrix
