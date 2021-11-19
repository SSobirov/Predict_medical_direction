import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, log_loss, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV



df = pd.read_excel("prepared_data.xlsx")
Y = df.Y.to_numpy()
X = df.iloc[:,0:707].to_numpy()
#print(X)
#print(len(Y))
#print(len(df.iloc[:,0:38].to_numpy()))
#transformer = RobustScaler().fit(X)
#X = transformer.transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

start = time.time()
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = GaussianNB()
classifier.fit(x_train, y_train)
end = time.time()

y_pred = classifier.predict(x_test)
print("\nNaive Bayes:")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Training time:", end-start)


"""
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(x_train, y_train)
print(rf_random.best_params_)
"""

start = time.time()
rf = RandomForestClassifier(n_estimators = 200, random_state = 42)
rf.fit(x_train, y_train)
end = time.time()
y_pred = rf.predict(x_test)
print("\nRandomForestClassifier:")
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training time:", end-start)
#confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)


start = time.time()
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
end = time.time()
y_pred = clf.predict(x_test)
print("\nDecisionTreeClassifier:")
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training time:", end-start)




start = time.time()
logisticRegr = LogisticRegression(solver = "liblinear")
logisticRegr.fit(x_train, y_train)
end = time.time()
score = logisticRegr.score(x_test, y_test)
y_pred = logisticRegr.predict(x_test)
print("\nLogisticRegression:")
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training time:", end-start)



# make predictions using xgboost random forest for classification
from xgboost import XGBRFClassifier
# define the model
start = time.time()
model = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2, eval_metric='mlogloss')
# fit the model on the whole dataset
model.fit(x_train, y_train)
end = time.time()
# make a prediction
y_pred = model.predict(x_test)
# summarize the prediction

print("\nXGBRFClassifier:")
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training time:", end-start)



from sklearn.ensemble import GradientBoostingClassifier
start = time.time()
model = GradientBoostingClassifier()
model.fit(x_train, y_train)
end=time.time()
y_pred = model.predict(x_test)
print("\nGradientBoostingClassifier:")
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training time:", end-start)
"""
"""
from sklearn.ensemble import HistGradientBoostingClassifier
start = time.time()
model = HistGradientBoostingClassifier(max_bins=255, max_iter=100)
model.fit(x_train, y_train)
end = time.time()
y_pred = model.predict(x_test)
print("\nHistGradientBoostingClassifier:")
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training time:", end-start)



from lightgbm import LGBMClassifier
start = time.time()
model = LGBMClassifier()
model.fit(x_train,y_train)
end = time.time()
#filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))
y_pred = model.predict(x_test)
print("\nLGBMClassifier:")
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training time:", end-start)



from catboost import CatBoostClassifier
start = time.time()
catboost_model = CatBoostClassifier(verbose=0, n_estimators=100)
catboost_model.fit(x_train, y_train)
end = time.time()
y_pred = catboost_model.predict(x_test)
print("\nCatBoostClassifier:")
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training time:", end-start)



from sklearn.svm import SVC
start = time.time()
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(x_train, y_train)
end = time.time()
y_pred = svm_model_linear.predict(x_test)
print("\nSVC:")
print("f1_score:", f1_score(y_test, y_pred, average = "weighted"))
print("f1_score for each class:", f1_score(y_test, y_pred, average = None))
print("Kappa score:", cohen_kappa_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training time:", end-start)

