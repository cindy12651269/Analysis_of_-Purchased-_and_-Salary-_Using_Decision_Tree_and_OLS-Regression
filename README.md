# Analysis of 'Purchased' and 'Salary' Using Decision Tree and OLS Regression

This project demonstrates two types of analysis on a dataset: classification and regression. The dataset is sourced from an online repository and contains information about individuals, including their country, age, salary, and whether they made a purchase. The analyses performed are:

1. **Classification of 'Purchased'**: Predicting whether an individual made a purchase based on their country, age, and salary using a decision tree classifier.
2. **Regression of 'Salary'**: Predicting an individual's salary based on their country, age, and purchase status using an ordinary least squares (OLS) regression model.

## Dataset

The dataset used in this project is located at: [Data.csv](https://raw.githubusercontent.com/MachineLearningLiuMing/scikit-learn-primer-guide/master/Data.csv)

## Prerequisites

To run the code, you need the following Python libraries installed:

- `pandas`
- `scikit-learn`
- `statsmodels`
- `numpy`

You can install them using pip:

```bash
pip install pandas scikit-learn statsmodels numpy
```

## Code Description

### Data Preprocessing

- **Load the dataset**: The dataset is loaded from the provided URL.
- **Handle missing values**: Missing values in the 'Salary' and 'Age' columns are filled with the mean values for each country.
- **Label encoding**: The categorical features 'Purchased' and 'Country' are converted to numerical values using label encoding.

### Analysis 1: Classification of 'Purchased'

- **Split the data**: The data is split into a training set (first 7 rows) and a testing set (last 3 rows).
- **Train a Decision Tree Classifier**: The classifier is trained on the training set to predict the 'Purchased' label.
- **Predict on the test set**: The classifier predicts the 'Purchased' labels for the test set.
- **Evaluate the model**: The accuracy of the classifier is calculated and printed.

### Analysis 2: Regression of 'Salary'

- **Split the data**: The data is split into a training set (first 7 rows) and a testing set (last 3 rows).
- **Train an OLS Regression Model**: The model is trained on the training set to predict the 'Salary'.
- **Predict on the test set**: The regression model predicts the 'Salary' values for the test set.
- **Evaluate the model**: The mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE) of the regression model are calculated and printed.

## Sample Code
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import numpy as np

# Load the dataset
source = 'https://raw.githubusercontent.com/MachineLearningLiuMing/scikit-learn-primer-guide/master/Data.csv'
df = pd.read_csv(source, sep=',', encoding='utf-8', engine='python')

# Fill missing values in 'Salary' and 'Age' with the mean values for each country
df['Salary'] = df.groupby('Country')['Salary'].transform(lambda x: x.fillna(x.mean()))
df['Age'] = df.groupby('Country')['Age'].transform(lambda x: x.fillna(x.mean()))

# Label encoding for 'Purchased' and 'Country'
label_encoder = LabelEncoder()
df['Purchase_label'] = label_encoder.fit_transform(df['Purchased'])
df['Country_label'] = label_encoder.fit_transform(df['Country'])

# Split the data into training (first 7 rows) and testing (last 3 rows) sets
train = df[:7]
y_train_clf = train['Purchase_label']
X_train_clf = train[['Age', 'Salary', 'Country_label']]

test = df[-3:]
y_test_clf = test['Purchase_label']
X_test_clf = test[['Age', 'Salary', 'Country_label']]

# Analysis 1: Classify 'Purchased' using 'Country', 'Age', 'Salary'
# Create and train a Decision Tree classifier
clf_model = DecisionTreeClassifier()
clf_model.fit(X_train_clf, y_train_clf)

# Predict 'Purchased' for the last three rows
y_pred_clf = clf_model.predict(X_test_clf)
predicted_purchased = label_encoder.inverse_transform(y_pred_clf)

print('Purchase Prediction:', predicted_purchased)

# Evaluate the accuracy of the classification model
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print("Accuracy:", accuracy)

# Analysis 2: Regress 'Salary' using 'Country', 'Age', 'Purchased'
# Prepare data for regression
y_train_reg = train['Salary']
X_train_reg = train[['Age', 'Country_label', 'Purchase_label']]

y_test_reg = test['Salary']
X_test_reg = test[['Age', 'Country_label', 'Purchase_label']]

# Create and train an OLS regression model
reg_model = sm.OLS(y_train_reg, X_train_reg).fit()
print(reg_model.summary())  # Model statistics

# Predict 'Salary' for the last three rows
y_pred_reg = reg_model.predict(X_test_reg)
print('Predicted Salary:', y_pred_reg)

# Evaluate the regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)


