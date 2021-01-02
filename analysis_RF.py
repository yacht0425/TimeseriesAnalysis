# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 08:38:53 2020

@author: yacht
"""


import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn import datasets
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

####################################################
#/                                                /#
#/ Read data and introduce "pandas" and "seaborn" /#
#/                                                /#
####################################################
############## Read data ###############
df = pd.read_csv('C:\\Users\\yacht\\Dropbox\\Skills\\2020-master\\2020-master\\AirPassenger.csv')
#df = df.drop('Unnamed: 3',axis=1)
#df.columns=['Date code','Date','Expenses']
#df = df.drop([0,1,2,3,4,5,6,7,8]).reset_index(drop=True)
#df = df.iloc[0:50,:]

############## Basic information ###############
print(df)
#print(df.head()) #initial 5 data
#print(df.tail(10)) #last 5 data
print(df.info()) #data type etc.
#print(df.describe()) #basic statistical information
#print(df['survived'].value_counts()) #extract data having a header 'survive' and count

############## Visualize using seaborn ###############
#sb.barplot(x='Year',y='AirPassenger',data=df)
#sb.countplot('survived', hue='pclass',data=df)
#sb.countplot('survived', hue='sex',data=df)
#sb.countplot('survived', hue='embarked',data=df)

####################################################
#/                                                /#
#/               Pre processing                  /#
#/                                                /#
####################################################
############## Edit data frame ###############
Year_month = []
j=0
k=194900
for _ in range((1961-1949)*12):
    j+=1
    Year_month.append(k+j)
    if(j==12):
        k=k+100
        j=0
        
Year_month = np.array(Year_month).reshape(-1,1)
#print(Year_month)
df['Date'] = Year_month    
df['Date'] = df['Date'].astype(str)
df['Date'] = pd.to_datetime(df['Date'],format='%Y%m')
df['UNIX_TIME'] = df['Date'].astype('int64').values//10**9
temp = df['AirPassenger'].values
temp2 = []
temp3 = []
temp4 = []
temp5 = []
temp6 = []
temp7 = []
temp8 = []
temp9 = []
temp10 = []


for i in range(len(temp)):
    if(i==0):
        temp2.append(0)
        continue
    temp2.append(temp[i-1])

temp2 = np.array(temp2)    

for i in range(len(temp)):
    if(i==0):
        temp3.append(0)
        continue
    if(i==1):
        temp3.append(0)
        continue
    temp3.append(temp[i-2])

temp3 = np.array(temp3)    

for i in range(len(temp)):
    if(i<12):
        temp4.append(0)
        continue
    temp4.append(temp[i-12])

temp4 = np.array(temp4)    

for i in range(len(temp)):
    if(i<24):
        temp5.append(0)
        continue
    temp5.append(temp[i-24])

temp5 = np.array(temp5)    

for i in range(len(temp)):
    if(i<36):
        temp6.append(0)
        continue
    temp6.append(temp[i-36])

temp6 = np.array(temp6)    

for i in range(len(temp)):
    if(i<48):
        temp7.append(0)
        continue
    temp7.append(temp[i-48])

temp7 = np.array(temp7)    

for i in range(len(temp)):
    if(i<60):
        temp8.append(0)
        continue
    temp8.append(temp[i-60])

temp8 = np.array(temp8)    


for i in range(len(temp)):
    if(i<72):
        temp9.append(0)
        continue
    temp9.append(temp[i-72])

temp9 = np.array(temp9)    

for i in range(len(temp)):
    if(i<84):
        temp10.append(0)
        continue
    temp10.append(temp[i-84])

temp10 = np.array(temp10)    


df['PreAirPassenger'] = temp2.reshape(-1,1)
df['PrePreAirPassenger'] = temp3.reshape(-1,1)
df['OneYearPreAirPassenger'] = temp4.reshape(-1,1)
df['TwoYearPreAirPassenger'] = temp5.reshape(-1,1)
df['ThreeYearPreAirPassenger'] = temp6.reshape(-1,1)
df['FourYearPreAirPassenger'] = temp7.reshape(-1,1)
df['FiveYearPreAirPassenger'] = temp8.reshape(-1,1)
df['SixYearPreAirPassenger'] = temp9.reshape(-1,1)
df['SevenYearPreAirPassenger'] = temp10.reshape(-1,1)
df['Month_square'] = df['Month']**2

print(df)

'''
############## Plot a figure X:Date, Y:Expenses ###############
plt.figure(figsize=(18,10))
plt.plot(df['Date'],df['Expenses'])
plt.grid()
'''

####################################################
#/                                                /#
#/         Create machine learning model          /#
#/                                                /#
####################################################
############## Make machine learning model ###############
x_time = df['Date'].values
x = df[['OneYearPreAirPassenger','SixYearPreAirPassenger']].values #0.60
#x = df[['UNIX_TIME','OneYearPreAirPassenger','TwoYearPreAirPassenger','ThreeYearPreAirPassenger']].values #0.59
#x = df[['OneYearPreAirPassenger','TwoYearPreAirPassenger','ThreeYearPreAirPassenger']].values #0.58
#x = df[['Year','Month','UNIX_TIME','OneYearPreAirPassenger','TwoYearPreAirPassenger','ThreeYearPreAirPassenger']].values #0.59
#x = df[['Year','Month','UNIX_TIME','OneYearPreAirPassenger','TwoYearPreAirPassenger']].values #0.59
#x = df[['Year','Month','UNIX_TIME','OneYearPreAirPassenger']].values #0.56
#x = df[['Year','Month','UNIX_TIME','PreAirPassenger','PrePreAirPassenger']].values
#x = df[['Year','Month','UNIX_TIME','PreAirPassenger']].values
#x = df[['Year','Month','PreAirPassenger']].values
#x = df[['Year','Month']].values
y = df['AirPassenger'].values


x_time = x_time.reshape(-1,1)
y=y.reshape(-1,1)

############## Split data to test and training ###############
N = len(x)
N_train = round(len(x)*0.8)
N_test = N - N_train
x_time_train,x_time_test = x_time[:N_train],x_time[N_train:]
x_train,y_train = x[:N_train],y[:N_train]
x_test,y_test = x[N_train:],y[N_train:]


############## Choose a model of Logistic regression ###############
model = RandomForestRegressor(random_state=0)
model.fit(x_train,y_train)
y_test_pred = model.predict(x_test)

############## Calculate accuracy ###############
print('Train-set R^2: {:.2f}'.format(model.score(x_train,y_train)))
print('Test-set R^2: {:.2f}'.format(model.score(x_test,y_test)))

############## Show plot ###############
plt.figure(figsize=(10,8))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.title('Random forest regressior, X = OneYearPreY, SixYearsPreY')
plt.plot(x_time,y,color='gray',linestyle='dashdot',label='True value')
plt.plot(x_time_train,y_train,'+',label='Prediction training')
#plt.plot(x_time_train,y_train)
plt.plot(x_time_test,y_test_pred,label='Prediction test')
plt.xlabel('Time [year]')
plt.ylabel('Passenger [persons]')
plt.legend(loc='upper left', fontsize=12)
plt.grid()
