# -*- coding: utf-8 -*-
"""Salary_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MHe0XyIb0a_rjUwpThrjnTU26bcSbWb6
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pickle
#plt.rc("font", size=14)
#sns.set(style="white")
#sns.set(style ="whitegrid", color_codes=True)

df=pd.read_csv('C:/Users/Dipak Das/Desktop/Salary_prediction/Salary.csv')
df

#print(df.shape)

#print(list(df.columns))

df.isnull().sum()

df['LastWorkingDate']=df['LastWorkingDate'].replace(np.nan,'2017-12-31')

df.head()

#corr_matrix=df.corr()
#sns.heatmap(data=corr_matrix, annot= True)

Start_Date=''
End_Date=''
act_year=[]
date_join=df['Dateofjoining']
date_end=df['LastWorkingDate']
for i in range(len(date_join)):
  start=[]
  end=[]
  Start_Date=date_join[i]
  start.append(Start_Date.split('-'))
  start_year=int(start[0][0])
  start_months=int(start[0][1])
  start_days=int(start[0][2])
  End_Date=date_end[i]
  end.append(End_Date.split('-'))
  end_year=int(end[0][0])
  end_months=int(end[0][1])
  end_days=int(end[0][2])
  act_year.append(float(end_year-start_year)+float((end_months-start_months)/12)+float((end_days-start_days)/365))
#print(act_year)

#print(df[0:10000:10]['Age'].unique().shape)
#print(df[0:10000:10]['Education_Level'].unique().shape)
#print(df[0:10000:10]['Designation'].unique().shape)
#print(df[0:10000:10]['Total Business Value'].unique().shape)

data=pd.DataFrame()
data['Age']=df[0:10000:10]['Age']
data['Education_Level']=df[0:10000:10]['Education_Level']
data['Experience']=act_year[0:10000:10]
data['Designation']=df[0:10000:10]['Designation']
data['Total_Business_Value']=df[0:10000:10]['Total Business Value']
#print(data.head())

#print(data[0:]['Experience'].unique().shape)
#print(data['Experience'].value_counts())

bus_val=[]
exp_val=[]
for i in data[0:1000:100]['Total_Business_Value']:
  bus_val.append(i)
for i in data[0:1000:100]['Experience']:
  exp_val.append(i)
#plt.bar(exp_val, bus_val)
#plt.title('Plot of Experience vs Business Value')
#plt.xlabel('Experience')
#plt.ylabel('Total Business Value')
#plt.show()

data['Designation'].value_counts()

#sns.countplot(x='Age', data=data)
#plt.show()
#plt.savefig('count_1_plot')

#sns.countplot(x='Designation', data=data)
#plt.show()
#plt.savefig('count_2_plot')

y=pd.DataFrame()
y['Salary']=df[0:10000:10]['Salary']
#print(y)

p1=[]
p2=[]
for i in data[0:1000:75]['Experience']:
  p1.append(i)
for j in y[0:1000:75]['Salary']:
  p2.append(j)
#plt.bar(p1,p2,width=0.1)
#plt.title('Plot of Experience vs Salary')
#plt.xlabel('Experience')
#plt.ylabel('Salary')
#plt.show()

#print(data.isnull().sum())
#print(y.isnull().sum())

categorical_var='Education_Level'

cat_list = 'var'+'_'+categorical_var
cat_list = pd.get_dummies(data[categorical_var], prefix=categorical_var)
data1 = data.join(cat_list)
data = data1

data_vars = data.columns.values.tolist()
#print(data_vars)

to_keep = [i for i in data_vars if i not in categorical_var]
to_keep

data_final = data[to_keep]
data_final.columns.values
#print(data_final)

#sns.set(rc={'figure.figsize':(11, 8)})
#sns.distplot(y['Salary'], bins= 30)
#plt.show()

#matrix=data_final.corr()
#sns.heatmap(data=matrix, annot= True)

#plt.scatter(data_final[0:]['Experience'],y)
#m, b=np.polyfit(data_final[0:]['Experience'],y,1)
#plt.plot(data_final[0:]['Experience'],m*data_final[0:]['Experience']+b,color='red')
#plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_final,y,test_size=0.4,random_state=42)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

model.coef_

model.intercept_

y_pred_data=model.predict(data_final)
# print(y_pred_data)

#plt.scatter(data_final[0:]['Experience'],y_pred_data)

#print("The equation of the multivariate linear regression is: ")
col_vars=list(data_final.columns.values)
#print(col_vars)

model.coef_[0][1]

x_eq=''
for i in range(len(col_vars)):
  x_eq+=str(model.coef_[0][i])+"*"+col_vars[i]+"+"
x_eq=x_eq+str(int(model.intercept_))
#print(x_eq)

y_pred=model.predict(x_test)
#print("Two Samply Predicted salaries on the test set y_test is: ")
#print(int(y_pred[0]))
#print(int(y_pred[1]))

print("Estimated Salary for Rajiv, age 22 and having 3 years of experience is: ", model.predict([[22, 3, 3, 200000, 0, 0, 1]])[0][0].round(2))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

score= r2_score(y_test, y_pred)
print("r2 score is ", score)
# y_truth= model.predict(x_train)
absolute_error= mean_absolute_error(y_test, y_pred)
mean_error= mean_squared_error(y_test, y_pred)
print("Mean absolute error is ", absolute_error)
print("Mean squared error is ", mean_error)


pickle.dump(model, open('model.pkl','wb'))

model_pickle = pickle.load( open('model.pkl','rb'))
print(model_pickle.predict([[22, 3, 3, 200000, 0, 0, 1]]))
