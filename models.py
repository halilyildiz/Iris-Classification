# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:24:43 2020

@author: halil
"""
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

#%% Veriyi eğiti ve test olarak parçalama
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


#%% SVC

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import  confusion_matrix


svc1 = SVC(kernel = "rbf")
svc1.fit(x_train,y_train)

svc1_pred = svc1.predict(x_test)
print("SVC-Default")
print(classification_report(y_test, svc1_pred)) 
print(confusion_matrix(y_test,svc1_pred))

#%% SVC-2

svc2 = SVC(kernel = "poly")
svc2.fit(x_train,y_train)

svc2_pred = svc2.predict(x_test)
print("SVC-kernel(Poly)")
print(classification_report(y_test, svc2_pred)) 
print(confusion_matrix(y_test,svc2_pred))

#%% SVC-3

svc3 = SVC(kernel = "linear")
svc3.fit(x_train,y_train)

svc3_pred = svc3.predict(x_test)
print("SVC-kernel(linear)")
print(classification_report(y_test, svc3_pred)) 
print(confusion_matrix(y_test,svc3_pred))

#%% Decission Tree  
from sklearn.tree import DecisionTreeClassifier

dt1 = DecisionTreeClassifier()
dt1.fit(x_train,y_train)

dt1_pred = dt1.predict(x_test)

print("Decission Tree-Default")
print(classification_report(y_test, dt1_pred)) 
print(confusion_matrix(y_test,dt1_pred))

#%% Decission Tree-2

dt2 = DecisionTreeClassifier(criterion = "entropy")
dt2.fit(x_train,y_train)

dt2_pred = dt2.predict(x_test)

print("Decission Tree-criterion(entropy)")
print(classification_report(y_test, dt2_pred)) 
print(confusion_matrix(y_test,dt2_pred))

#%% Decission Tree-3

dt3 = DecisionTreeClassifier(splitter = "random")
dt3.fit(x_train,y_train)

dt3_pred = dt3.predict(x_test)

print("Decission Tree-splitter(random)")
print(classification_report(y_test, dt3_pred)) 
print(confusion_matrix(y_test,dt3_pred))

#%% KNN-1 Modeli

from sklearn.neighbors import KNeighborsClassifier

knn1 = KNeighborsClassifier()
knn1.fit(x_train,y_train)
 
knn1_pred = knn1.predict(x_test)

print("KNN-Default")
print(classification_report(y_test, knn1_pred)) 
print(confusion_matrix(y_test,knn1_pred))

#%% KNN-2 

from sklearn.neighbors import KNeighborsClassifier

knn2 = KNeighborsClassifier(n_neighbors = 10)
knn2.fit(x_train,y_train)
 
knn2_pred = knn1.predict(x_test)

print("KNN-2")
print(classification_report(y_test, knn2_pred)) 
print(confusion_matrix(y_test,knn2_pred))

#%% KNN-3 

from sklearn.neighbors import KNeighborsClassifier

knn3 = KNeighborsClassifier(weights = "distance")
knn3.fit(x_train,y_train)
 
knn3_pred = knn3.predict(x_test)

print("KNN-3")
print(classification_report(y_test, knn3_pred)) 
print(confusion_matrix(y_test,knn3_pred))

#%% Random Forest 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

rf_pred = rf.predict(x_test)

print("Random Forest")
print(classification_report(y_test, rf_pred)) 
print(confusion_matrix(y_test,rf_pred))

#%% Ada Boost Classifier Modeli
from sklearn.ensemble import AdaBoostClassifier

abc1 = AdaBoostClassifier()
abc1.fit(x_train,y_train)

abc1_pred = abc1.predict(x_test)

print("Ada Boost Classifier")
print(classification_report(y_test, abc1_pred)) 
print(confusion_matrix(y_test,abc1_pred))

#%% Ada Boost Classifier -2 Modeli
svc=SVC(probability=True, kernel='linear')

# Create adaboost classifer object
abc2 = AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

# Train Adaboost Classifer
abc2.fit(x_train, y_train)

#Predict the response for test dataset
abc2_pred = abc2.predict(x_test)

print("Ada Boost Classifier-2 ")
print(classification_report(y_test, abc1_pred)) 
print(confusion_matrix(y_test,abc1_pred))

#%% BaggingClassifier

from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier()
bc.fit(x_train,y_train)

bc_pred = bc.predict(x_test)

print("BaggingClassifier")
print(classification_report(y_test, bc_pred)) 
print(confusion_matrix(y_test, bc_pred))

#%% BaggingClassifier-2
svc=SVC(probability=True, kernel='linear')

# Create adaboost classifer object
bc2 = BaggingClassifier(n_estimators=50, base_estimator=svc)

# Train Adaboost Classifer
bc2.fit(x_train, y_train)

#Predict the response for test dataset
bc2_pred = bc2.predict(x_test)

print("BaggingClassifier-2 ")
print(classification_report(y_test, bc2_pred)) 
print(confusion_matrix(y_test, bc2_pred))

#%% VotingClassifier
from sklearn.ensemble import VotingClassifier

svc = SVC()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

vc = VotingClassifier(estimators=[('svc', svc), ('dt', dt), ('knn', knn)],voting='hard')

vc.fit(x_train, y_train)

vc_pred = vc.predict(x_test)

print("VotingClassifier")
print(classification_report(y_test, vc_pred)) 
print(confusion_matrix(y_test, vc_pred))



