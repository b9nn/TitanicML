import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# notes about the dataset
# PClass --> economic class 1 (upper), 2 (middle), 3 (lower)
# SibSp --> siblings or spouses onboard: int
# Parch --> parents/children onboard: int

# preprocessing 

data['Age'] = data['Age'].fillna(data['Age'].median()) # fill missing ages with the median age
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0]) # mode method fills with most frequent departure location
data['Sex'] = data['Sex'].fillna(data['Sex'].mode()[0]) # option to either put unknown or most common
data['Name'] = data['Name'].fillna('Unknown') # fill unknown names with "unknown"

# encode strings --> assigns each sex and embarkation data their own value
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])

# create column for optimized training
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1  # +1 includes the passenger themselves

# define axis
X = data[['FamilySize', 'Sex', 'Age', 'Fare', 'Embarked']]
y = data['Survived']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # %20 to testing, %80 to training

# grid search for hyperparameters 

# n_estimators --> number or trees

parameter_grid = {'n_estimators':[5, 50, 100, 150, 200, 250, 300], 'max_depth': [None, 5, 10, 15, 20], 'min_samples_split':[2, 5, 10, 15, 20]} # define all parameters to be tested on by gridsearch

# train random forest model (combines all decisoin trees and returns regression of all trees)
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, parameter_grid, cv=5, scoring='accuracy') # finds best hyperparameters to tune the rf
grid_search.fit(X_train, y_train) # train model

best_rf = grid_search.best_estimator_ # best model

y_pred = best_rf.predict(X_test) # predict survivability on the x tests
y_probability = best_rf.predict_proba(X_test)[:, 1] # assingn probabilites ot each class
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Analysis

falsep_rate, truep_rate, threshold = roc_curve(y_test, y_probability) # how well did the binary classifier do?
roc_auc = auc(falsep_rate, truep_rate) # auc gives overall deiscriminative ability

# plotting roc and auc
plt.figure(figsize=(6, 4))
plt.plot(falsep_rate, truep_rate, label=f'AUC = {roc_auc:.2f}', color='blue')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# plotting features
importances = best_rf.feature_importances_
features = X.columns
plt.figure(figsize=(6,4))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()