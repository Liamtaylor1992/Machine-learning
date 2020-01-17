import matplotlib.pyplot as plt
import numpy as np
import pandas
import time

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.ensemble import VotingClassifier

filename = 'bruteforcedataset.csv'
CSV = pandas.read_csv(filename)

with open(filename, 'r') as csv:
                    first_line = csv.readline()
                    your_data = csv.readlines()

ncol = first_line.count(',')

Array = CSV.values
X = Array[:,0:ncol]
y = Array[:,ncol]

#X_new = SelectKBest(mutual_info_classif, k=10).fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Random Forest

startrf = time.time()
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
RF_accuracy = accuracy_score(y_test, pred_rfc)
endrf = time.time()


Support Vector Machine

startsv = time.time()
SVM = svm.SVC(gamma='auto')
SVM = SVM.fit(X_train, y_train)
pred_SVM = SVM.predict(X_test)
SVM_accuracy = accuracy_score(y_test, pred_SVM)
endsv = time.time()


#NEURAL NETWORK MODEL

startnn = time.time()
mlpc=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)
total_sum = 0
for i in range(10):
    mlpc.fit(X_train, y_train)
    pred_mlpc = mlpc.predict(X_test)
    MLPC_accuracy = accuracy_score(y_test, pred_mlpc)
    total_sum += MLPC_accuracy
MLPC_accuracyaverage = total_sum/10
endnn = time.time()



#Gaussian Naive Bayes

startgn = time.time()
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
pred_gnb = gnb.predict(X_test)
gnb_accuracy = accuracy_score(y_test, pred_gnb)
endgn = time.time()

#Logistic Regression

startlr = time.time()
lr = LogisticRegression(solver='lbfgs', max_iter=500)
lr = lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, pred_lr)
endlr = time.time()


#NORMAL LINEAR DISCRIMINANT ANALYSIS

startld = time.time()
lda = LinearDiscriminantAnalysis()
lda = lda.fit(X_train, y_train)
pred_lda = lda.predict(X_test)
lda_accuracy = accuracy_score(y_test, pred_lda)
endld = time.time()

#Ensemble VotingClassifier

startes = time.time()
model = VotingClassifier(estimators=[('rfc', rfc), ('SVM', SVM), ('mlpc', mlpc), ('gnb', gnb), ('lr', lr), ('lda', lda)])
model.fit(X_train,y_train)
voting = model.score(X_test,y_test)
endes = time.time()


#Printing outputs

print filename

print 'RANDOM FOREST'
print(classification_report(y_test, pred_rfc))
print 'Confusion matrix'
print(confusion_matrix(y_test, pred_rfc))
print 'Random Forest Accuracy', RF_accuracy
rftime = endrf - startrf
print 'execution time', rftime

print '\n'
print 'SUPPORT VECTOR MACHINE'
print(classification_report(y_test, pred_SVM))
print 'Confusion matrix'
print(confusion_matrix(y_test, pred_SVM))
print 'Support Vector Machine Accuracy', SVM_accuracy
svmtime = endsv - startsv
print 'execution time', svmtime

print'\n'
print 'NEURAL NETWORK MODEL'
print(classification_report(y_test, pred_mlpc))
print 'Confusion matrix'
print(confusion_matrix(y_test, pred_mlpc))
print 'Neural network model Average Accuracy is', MLPC_accuracyaverage
nntime = endnn - startnn
nntime = nntime / 10
print 'Average execution time', nntime
print'\n'

print 'GAUSSIAN NAIVE BAYES'
print(classification_report(y_test, pred_gnb))
print 'Confusion matrix'
print(confusion_matrix(y_test, pred_gnb))
print 'Gaussian Naive Bayes Accuracy', gnb_accuracy
gnbtime = endgn - startgn
print 'execution time', gnbtime

print'\n'
print 'LOGISTIC REGRESSION'
print(classification_report(y_test, pred_lr))
print 'Confusion matrix'
print(confusion_matrix(y_test, pred_lr))
print 'Logistic Regression Accuracy', lr_accuracy
lrtime = endlr - startlr
print 'execution time', lrtime 

print'\n'
print 'NORMAL LINEAR DISCRIMINANT ANALYSIS'
print(classification_report(y_test, pred_lda))
print 'Confusion matrix'
print(confusion_matrix(y_test, pred_lda))
print 'Normal Linear Discriminant Analysis Accuracy', lda_accuracy
ldatime = endld - startld
print 'execution time', ldatime

print '\n'
print 'Ensemble'
print 'Ensemble accuracy', voting
print 'Ensemble execution time', endes - startes
