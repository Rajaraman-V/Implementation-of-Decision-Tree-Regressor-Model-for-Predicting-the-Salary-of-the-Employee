# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries, load the Salary dataset, and convert categorical columns such as Position into numerical values using Label Encoding.
2. Select Position and Level as input features (X) and Salary as the target variable (y). Split the dataset into training and testing sets.
3. Create the DecisionTreeRegressor model and train it using the training dataset.
4. Predict salary values for the test dataset, calculate the R² score to measure performance, and visualize the decision tree using plot_tree()

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RAJARAMAN V
RegisterNumber:  212223110038
```
```
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
df = pd.read_csv("Salary.csv")

print("Dataset Preview:")
print(df.head())

X = df[["Level"]]   
y = df["Salary"]              
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE  :", mean_absolute_error(y_test, y_pred))
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2_score(y_test, y_pred))
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Level"],
    filled=True
)
plt.title("Decision Tree Regressor for Employee Salary Prediction")
plt.show()
new_exp = [[5]]  
predicted_salary = model.predict(new_exp)
print("\nPredicted Salary for 5 years experience:", predicted_salary[0])
```

## Output:
<img width="1252" height="221" alt="image" src="https://github.com/user-attachments/assets/adbb69a0-3583-4eec-a5da-f4b7813ee6bc" />
<img width="1089" height="639" alt="image" src="https://github.com/user-attachments/assets/a661d6e0-ce9e-4258-9e97-fba606a9d8b3" />
<img width="844" height="36" alt="image" src="https://github.com/user-attachments/assets/9a332b53-9246-48db-aee9-4aa508f42eda" />





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
