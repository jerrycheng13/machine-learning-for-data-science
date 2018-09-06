import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix # use it from sklearn to create confusion_matrix and compute false positive and false negative rate
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv('adult-new.data', sep=",", header=None, skipinitialspace=True)
test = pd.read_csv('adult-new.test', sep=",", header=None, skipinitialspace=True)
train.shape
test.shape

X = train.append(test)
X.shape
X.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
X['income'] = X['income'].map({'>50K': 1, '<=50K': 0})
X = pd.get_dummies(X)
X.shape

label = X[['income']]
feature = X.drop(['income'], axis=1)
X_train = feature[0:32561]
X_test = feature[32561:]
y_train = label[0:32561]
y_test = label[32561:]

# sex
X_Male = X[32561:].loc[X[32561:]['sex_Male'] == 1]
X_Female = X[32561:].loc[X[32561:]['sex_Female'] == 1]
y_Male_test = X_Male[['income']]
y_Female_test = X_Female[['income']]
X_Male_test = X_Male.drop(['income'], axis=1)
X_Female_test = X_Female.drop(['income'], axis=1)


# linearSVM(affine classifier) # use LinearSVC and MinMaxScaler from sklearn
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm = LinearSVC(C=10)
svm.fit(X_train_scaled, y_train)
print("Train set score: {:.4f}".format(svm.score(X_train_scaled, y_train)))
print("Test set score: {:.4f}".format(svm.score(X_test_scaled, y_test)))
# scale
X_Male_test_scaled = scaler.transform(X_Male_test)
X_Female_test_scaled = scaler.transform(X_Female_test)
## Male
print("Confusion matrix:\n{}".format(confusion_matrix(y_Male_test, svm.predict(X_Male_test_scaled))))
svm_male = confusion_matrix(y_Male_test, svm.predict(X_Male_test_scaled))
print("False Positive rate of Male: {:.4f}".format(svm_male[0,1]/(svm_male[0,0]+svm_male[0,1])))
print("False Negative rate of Male: {:.4f}".format(svm_male[1,0]/(svm_male[1,0]+svm_male[1,1])))
## Female
print("Confusion matrix:\n{}".format(confusion_matrix(y_Female_test, svm.predict(X_Female_test_scaled))))
svm_female = confusion_matrix(y_Female_test, svm.predict(X_Female_test_scaled))
print("False Positive rate of Female: {:.4f}".format(svm_female[0,1]/(svm_female[0,0]+svm_female[0,1])))
print("False Negative rate of Female: {:.4f}".format(svm_female[1,0]/(svm_female[1,0]+svm_female[1,1])))


# random forest # use RandomForestClassifier from sklearn
forest = RandomForestClassifier(n_estimators=50, random_state=0)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.4f}".format(forest.score(X_train, y_train))) 
print("Accuracy on test set: {:.4f}".format(forest.score(X_test, y_test)))
## Male
print("Confusion matrix:\n{}".format(confusion_matrix(y_Male_test,forest.predict(X_Male_test))))
forest_male = confusion_matrix(y_Male_test,forest.predict(X_Male_test))
print("False Positive rate of Male: {:.4f}".format(forest_male[0,1]/(forest_male[0,0]+forest_male[0,1])))
print("False Negative rate of Male: {:.4f}".format(forest_male[1,0]/(forest_male[1,0]+forest_male[1,1])))
## Female
print("Confusion matrix:\n{}".format(confusion_matrix(y_Female_test, forest.predict(X_Female_test))))
forest_female = confusion_matrix(y_Female_test, forest.predict(X_Female_test))
print("False Positive rate of Female: {:.4f}".format(forest_female[0,1]/(forest_female[0,0]+forest_female[0,1])))
print("False Negative rate of Female: {:.4f}".format(forest_female[1,0]/(forest_female[1,0]+forest_female[1,1])))


# neural network # use MLPClassifier from sklearn
mlp = MLPClassifier(max_iter=1000, alpha=0.00001, random_state=0)
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.4f}".format(mlp.score(X_train, y_train))) 
print("Accuracy on test set: {:.4f}".format(mlp.score(X_test, y_test)))
## Male
print("Confusion matrix:\n{}".format(confusion_matrix(y_Male_test, mlp.predict(X_Male_test))))
mlp_male = confusion_matrix(y_Male_test, mlp.predict(X_Male_test))
print("False Positive rate of Male: {:.4f}".format(mlp_male[0,1]/(mlp_male[0,0]+mlp_male[0,1])))
print("False Negative rate of Male: {:.4f}".format(mlp_male[1,0]/(mlp_male[1,0]+mlp_male[1,1])))
## Female
print("Confusion matrix:\n{}".format(confusion_matrix(y_Female_test, mlp.predict(X_Female_test))))
mlp_female = confusion_matrix(y_Female_test, mlp.predict(X_Female_test))
print("False Positive rate of Female: {:.4f}".format(mlp_female[0,1]/(mlp_female[0,0]+mlp_female[0,1])))
print("False Negative rate of Female: {:.4f}".format(mlp_female[1,0]/(mlp_female[1,0]+mlp_female[1,1])))






