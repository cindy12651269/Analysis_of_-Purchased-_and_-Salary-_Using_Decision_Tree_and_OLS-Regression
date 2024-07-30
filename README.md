# Analysis_of_-Purchased-_and_-Salary-_Using_Decision_Tree_and_OLS-Regression
This project demonstrates two types of analysis on a dataset: classification and regression. The dataset is sourced from an online repository and contains information about individuals, including their country, age, salary, and whether they made a purchase. The analyses performed are:

1. Classification of 'Purchased': Predicting whether an individual made a purchase based on their country, age, and salary using a decision tree classifier.

2. Regression of 'Salary': Predicting an individual's salary based on their country, age, and purchase status using an ordinary least squares (OLS) regression model.

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

```python
