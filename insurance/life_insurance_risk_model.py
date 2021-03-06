# -*- coding: utf-8 -*-
"""Life Insurance Risk Model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Qd3eiX9C18wTPE55Ru5ycaE-oeX3TCuh

# **DATA SCIENCE TEAM 2**

Anade Davis - Data Science Manager

Ragavendhra Ramanan - Project Lead

Hafizah Ab Rahim - Data Scientist

Brandon Oppong - Antwi -Data Engineer

Raques McGill - Data Scientist

Ivy Hsu - Data Scientist

# Life Insurance risk model risk model that users can input certain data points (ex. Age, Smoker, Accidents in last 5 years) to offer a insurance rate immediately

# **Sources:**


https://www.ssa.gov/oact/STATS/table4c6.html

https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99

https://medium.com/swlh/a-machine-learning-web-app-to-predict-insurance-claim-charges-533f6e8179f1

##**Import Libraries**
"""

import pandas as pd #pandas is a software library written for the Python programming language for data manipulation and analysis.
import seaborn as sns #seaborn is a software library for Python that Matplotlib to visualize random distributions,
from matplotlib import pyplot as plt #matplotlib is a software library for Python that creates statisticals graphs/plots of our data
import numpy as np #numpy is a software library written for Python so that we can linear algebra as well as perform mathematical operations on matrices.
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

"""##**Upload Data**"""

df = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv') #read and upload our raw data

"""##**Data Inspection**"""

df.head() #prints a snapshot of the first five rows of our dataset

df.info() #gain important feature inofrmation abour our data

df.describe(include='all') # used to view some basic statistical details like percentile, mean, std etc. of our dataset

"""##**Data Wrangling**"""

dummy=pd.get_dummies(df[['sex','smoker','region']]) #convert categorical variable into dummy/indicator variables

dummy

df2 = pd.concat([df,dummy], axis=1) #combine the dummy and our original dataset into one

df2.head()

df2.drop(['sex','smoker','region','sex_male','smoker_no'], axis=1, inplace=True) #drop the variable features that we do not need

df2.head()

df2= df2.rename(columns={'sex_female':'gender', 'smoker_yes':'smoker','region_northeast':'northeast', 'region_northwest':'northwest', 'region_southeast':'southeast', 'region_southwest':'southwest'}) # 1-female, 0-male, 1-smoker, 0- non smoker

df2.head(3)

df2=df2[['age','bmi','children','gender','smoker','northeast','northwest','southeast','southwest','charges']]

df2.head(1)

"""##**Exploratory Data Analysis**

####Pearson Correlation
- is used to measure the strength and direction (+/-) of the **linear relationship** between two variables.
- gives a quick idea of the potential usefulness of features.
- only valid for **continuous data**, not appropriate for a **binary response variable**.
- correlation (values in the table) are between **0** (not linearly correlated) and **1** or **-1** (highly correlated).
"""

plt.figure(figsize=(12,8))
sns.heatmap(df2.corr(), annot=True);

"""####Scatterplot Matrix
- similar to correlation plot.
- shows all data as a grid or scatter plots of all features and the response variable.
- examine data directly in a **concise format**.
"""

# the scatterplot shows the relationship between numerical features and target/label class.
# As the bmi increases, the charges increases.
# 
sns.set_style(style='ticks')
sns.pairplot(df,hue='smoker'); #include a pairplot of our x and y variables

df2['age'].hist(bins=50, color='orange')
plt.xlabel('age')
plt.ylabel('charges');

df2['bmi'].hist(bins=50, color='orange')
plt.xlabel('bmi')
plt.ylabel('charges');

df2['children'].hist(bins= 25, color='orange')
plt.xlabel('children')
plt.ylabel('charges');

region_ = df.groupby('region').agg({'charges':np.mean})
region_.plot(kind='bar', colormap='Pastel2');

gender_ = df2.groupby('gender').agg({'charges':np.mean})
gender_.plot(kind='bar', colormap='GnBu');

smoker_ = df2.groupby('smoker').agg({'charges':np.mean})
smoker_.plot(kind='bar', colormap='Spectral');

"""##**Feature Selection**

####ANOVA F-test
- ANOVA stands for "analysis of variance".
- Used to test whether features are associated with a response
"""

features = df2.columns.tolist()
print(features)

X = df2[features].iloc[:,:-1].values
y = df2[features].iloc[:,-1].values

print(X.shape, y.shape)

[f_stat, f_p_value] = f_regression(X,y)

df_test = pd.DataFrame({'Feature':features[:-1], 'F-Score':f_stat, 'p-value': f_p_value})
df_test.sort_values('p-value')

"""##**Split Data & Feature Scaling**"""

#X1 = df2[['smoker','age', 'bmi']].values
#y1 = df2[features].iloc[:,-1].values

#print(X1.shape, y1.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state=0)

X_train.shape

X_test.shape

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train

X_test = scaler.transform(X_test)

"""##**Build Model**
######**Linear Regression**
"""

regressor = LinearRegression()
lm = regressor.fit(X_train, y_train)
lm_pred = lm.predict(X_test)

accuracy = regressor.score(X_test, y_test)
print('Accuracy of Linear Regression model = '+ str(round(accuracy,3)))

"""######**Random Forest Regression**"""

rf_reg = RandomForestRegressor(n_estimators=10, random_state=4)
rf = rf_reg.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

accuracy = rf_reg.score(X_test, y_test)
print('Accuracy of Random Forest model = '+ str(round(accuracy,3)))

"""##**Save Model**"""

# Save model - in binary mode
filename = 'Life insurance Model.pkl'
pickle.dump(rf_reg, open(filename, 'wb'))
