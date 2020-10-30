import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
CompanyData = pd.read_csv("D:\\ExcelR Data\\Assignments\\Decision+Tree\\Company_Data.csv")
CompanyData.columns
Le = preprocessing.LabelEncoder()
#convertong catogorical to numerical
CompanyData['shelveLoc']=Le.fit_transform(CompanyData['ShelveLoc'])
CompanyData['urban']=Le.fit_transform(CompanyData['Urban'])
CompanyData['us']=Le.fit_transform(CompanyData['US'])
#Droping
CompanyData.drop(["ShelveLoc"],inplace=True,axis=1)
CompanyData.drop(["Urban"],inplace=True,axis=1)
CompanyData.drop(["US"],inplace=True,axis=1)

CompanyData.head()
# converting float value to integer S we are converting float to catogorical data
bins=[-1,6,12,18]
CompanyData["Sales"]=pd.cut(CompanyData["Sales"],bins,labels=["lower","medium","high"])

CompanyData['Sales'].unique()
CompanyData.Sales.value_counts()
colnames = list(CompanyData.colnames)
predictors = colnames[1:11] # Input columns
target = colnames[0] # output columns

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(CompanyData,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
# help(DecisionTreeClassifier) for study


model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])
type(preds)
pd.Series(preds).value_counts()

pd.crosstab(test[target],preds)


np.mean(preds==test.Sales) # 0.7625
