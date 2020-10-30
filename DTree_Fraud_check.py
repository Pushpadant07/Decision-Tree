import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
Fraudcheck = pd.read_csv("D:\\ExcelR Data\\Assignments\\Decision+Tree\\Fraud_check.csv")
Fraudcheck.columns
Le = preprocessing.LabelEncoder()
Fraudcheck['undergrad']=Le.fit_transform(Fraudcheck['Undergrad'])
Fraudcheck['marital_Status']=Le.fit_transform(Fraudcheck['Marital_Status'])
Fraudcheck['urban']=Le.fit_transform(Fraudcheck['Urban'])
#Droping
Fraudcheck.drop(["Undergrad"],inplace=True,axis=1)
Fraudcheck.drop(["Marital_Status"],inplace=True,axis=1)
Fraudcheck.drop(["Urban"],inplace=True,axis=1)

Fraudcheck.head()
# converting float value to integer S we are converting float to catogorical data
bins=[-1,30000,100000]
Fraudcheck["Taxable_Income"]=pd.cut(Fraudcheck["Taxable_Income"],bins,labels=["Risky","Good"])

Fraudcheck['Taxable_Income'].unique()
Fraudcheck.Taxable_Income.value_counts()
colnames = list(Fraudcheck.columns)
predictors = colnames[1:6] # input columns
target = colnames[0] #output columns

# Splitting data into training and testing data set


from sklearn.model_selection import train_test_split
train,test = train_test_split(Fraudcheck,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)


model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])
type(preds)
pd.Series(preds).value_counts()

pd.crosstab(test[target],preds)


np.mean(preds==test.Taxable_Income) # 0.6333333333333333
