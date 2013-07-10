# Load up all the necessary packages.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc, auc_score
import pylab as pl
from numpy import append

#read in the data files
train_data = pd.read_csv("/Users/davidlogsdon/data_science/Homework_3/train1.csv")
test_data =  pd.read_csv("/Users/davidlogsdon/data_science/Homework_3/test1.csv")

#briefly combining both the training and testing data sets to create the full CountVectorizer
subset1 = train_data[['Insult', 'Comment']]
#subset2 = test_data.rename(columns={'id': 'Insult'})
subset2 = test_data[['Comment']]
subset3= subset1.append(subset2)

v = CountVectorizer()
Comments_train = v.fit_transform(subset3.Comment)
Insults_train = train_data.Insult
cv_array = Comments_train.toarray()[0:3947]

clf = LogisticRegression(C=.8).fit(cv_array, Insults_train)

Insults_test_predict = clf.predict(Comments_train.toarray()[3947:])
probas_Insults_test_predict_logistic = clf.predict_proba(Comments_train.toarray()[3947:])
df = pd.DataFrame(probas_Insults_test_predict_logistic, columns = ['a', 'b'])
df.to_csv("/Users/davidlogsdon/data_science/Homework_3/FINAL_PREDICTIONS.csv")

#note, not posting testing code, found naive bayes to be less effective
#a model and removed it from final submission