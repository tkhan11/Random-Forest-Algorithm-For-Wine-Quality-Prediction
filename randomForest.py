#Importing required packages.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


wine = pd.read_csv("./winequality-white.csv")

# See the number of rows and columns
print("Dataset Shape:", wine.shape)

# See the first five rows of the dataset
print(wine.head())

# Create Classification version of target variable
wine['quality'] = [1 if x >6 else 0 for x in wine['quality']]

# Separate feature variables and target variable
X = wine.drop(['quality'], axis = 1)
y = wine['quality']

print("\nGood (1) and Bad (0):\n",wine['quality'].value_counts())

# Apply Normalization
X = StandardScaler().fit_transform(X)
# print("After Applying normalization:\n", X)

#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("Dataset after splitting into train and test sets:\n",X_train.shape,X_test.shape,y_train.shape,y_test.shape,"\n")

rfc = RandomForestClassifier(n_estimators=100) # number of tress in the Random forest
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

#Let's see how our model performed
print(classification_report(y_test, pred_rfc),"\n")

#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_rfc),"\n")

accuracy = accuracy_score(y_test, pred_rfc)
print('Accuracy:' , accuracy)
