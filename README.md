# Telecom Churn Dataset

## Overview
The Telecom Churn Dataset contains information about a telecom company's customers and their behavior, which can be used to predict whether a customer will churn (leave the company) or not. This dataset is valuable for building predictive models to identify at-risk customers and devise retention strategies.

## Dataset Description
The dataset includes various customer attributes such as demographics, account information, and usage patterns. Each row represents a customer, and each column represents a customer attribute or a behavioral metric.

### Columns
- **customerID**: Unique identifier for each customer.
- **gender**: Customer's gender (Male, Female).
- **SeniorCitizen**: Indicates if the customer is a senior citizen (1, 0).
- **Partner**: Indicates if the customer has a partner (Yes, No).
- **Dependents**: Indicates if the customer has dependents (Yes, No).
- **tenure**: Number of months the customer has stayed with the company.
- **PhoneService**: Indicates if the customer has phone service (Yes, No).
- **MultipleLines**: Indicates if the customer has multiple lines (Yes, No, No phone service).
- **InternetService**: Customer's internet service provider (DSL, Fiber optic, No).
- **OnlineSecurity**: Indicates if the customer has online security (Yes, No, No internet service).
- **OnlineBackup**: Indicates if the customer has online backup (Yes, No, No internet service).
- **DeviceProtection**: Indicates if the customer has device protection (Yes, No, No internet service).
- **TechSupport**: Indicates if the customer has tech support (Yes, No, No internet service).
- **StreamingTV**: Indicates if the customer has streaming TV (Yes, No, No internet service).
- **StreamingMovies**: Indicates if the customer has streaming movies (Yes, No, No internet service).
- **Contract**: The contract term of the customer (Month-to-month, One year, Two year).
- **PaperlessBilling**: Indicates if the customer has paperless billing (Yes, No).
- **PaymentMethod**: The payment method used by the customer (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
- **MonthlyCharges**: The amount charged to the customer monthly.
- **TotalCharges**: The total amount charged to the customer.
- **Churn**: Indicates if the customer churned (Yes, No).

## Data Preparation
### Missing Values
Handle missing values by imputing them or removing the rows/columns with missing values.

### Encoding Categorical Variables
Convert categorical variables into numerical formats using techniques such as one-hot encoding or label encoding.

### Feature Scaling
Normalize or standardize numerical features to bring them to a common scale.

## Analysis and Modeling
### Exploratory Data Analysis (EDA)
- Analyze the distribution of features.
- Identify correlations between features.
- Visualize the data using plots to understand relationships and patterns.

### Predictive Modeling
- Split the data into training and testing sets.
- Train various machine learning models (e.g., Logistic Regression, Decision Trees, Random Forests, Gradient Boosting).
- Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

## Usage
1. Load the dataset into a Pandas DataFrame.
2. Perform EDA to understand the data.
3. Preprocess the data (handle missing values, encode categorical variables, scale features).
4. Split the data into training and testing sets.
5. Train machine learning models and evaluate their performance.
6. Use the best-performing model to make predictions on new data.

## Sample Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df1 = pd.read_csv(r"C:\Users\Administrator\OneDrive\Telecom Churn Dataset\churn_data.csv")
df2 = pd.read_csv(r"C:\Users\Administrator\OneDrive\Telecom Churn Dataset\customer_data.csv")
df3 = pd.read_csv(r"C:\Users\Administrator\OneDrive\Telecom Churn Dataset\internet_data.csv")

# Concat all dataset into one DataFrame

# Handle missing values
df4.dropna(inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df4.select_dtypes(include=['object']).columns:
    df4[col] = le.fit_transform(df4[col])

# Feature scaling
scaler = StandardScaler()
df[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['MonthlyCharges', 'TotalCharges']])

# Split the data
X = df4.drop('Churn', axis=1)
y = df4['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



**Feel free to modify the sections to better fit your specific dataset and use case. This README provides a comprehensive overview, including dataset description, preparation steps, analysis and modeling suggestions, usage instructions, and sample code.**
