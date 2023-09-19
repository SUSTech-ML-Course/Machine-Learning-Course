# There is one or more mistake(s) in the code. See if you can find it.

# 'os' module provides functions for interacting with the operating system 
import os

# 'Numpy' is used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np

# 'Pandas' is used for data manipulation and analysis
import pandas as pd

# 'Matplotlib' is a data visualization library for 2D and 3D plots, built on numpy
from matplotlib import pyplot as plt
# %matplotlib inline

# 'Seaborn' is based on matplotlib; used for plotting statistical graphics
import seaborn as sns

# 'SciPy' is used to perform scientific computations
import scipy.stats as stats

#
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# to suppress warnings
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('load_loans.csv')

print('Various info of dataset:\n')

print('df.head(n=10)\n', df.head(n=10))
print('\ndf.shape\n', df.shape)
print('\ndf.info()')
df.info()
print('\ndf.dtypes\n', df.dtypes)
print('\ndf.describe()\n', df.describe())
print('\ndf.describe(exclude=[np.number])\n', df.describe(exclude=[np.number]))

print('Various info of single variable:\n')

print('\ndf[\'Loan_Status\'].value_counts()\n', df['Loan_Status'].value_counts())
print('\ndf[\'Loan_Status\'].value_counts(normalize=True)\n', df['Loan_Status'].value_counts(normalize=True))

print('\nVarious plot:\n')
sns.countplot(x='Loan_Status', data=df, palette = 'Set1')
plt.show()

Credit_History=df['Credit_History'].value_counts(normalize=True)
Credit_History.plot.bar(title= 'Credit_History')
plt.show()

Credit_History=pd.crosstab(df['Credit_History'],df['Loan_Status'])
Credit_History.plot(kind="bar", stacked=True, figsize=(5,5))
plt.show()

print('Convert the data types:\n')

df['ApplicantIncome'] = df['ApplicantIncome'].astype('float64')
df = df.drop('Loan_ID',axis=1)
df.info()

print('\ndf.isnull().sum()\n', df.isnull().sum())

print('\nFill missing value:\n')

df['Gender'].fillna(df['Gender'].value_counts().idxmax(), inplace=True)
df['Married'].fillna(df['Married'].value_counts().idxmax(), inplace=True)
df['Dependents'].fillna(df['Dependents'].value_counts().idxmax(), inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].value_counts().idxmax(), inplace=True)

df["LoanAmount"].fillna(df["LoanAmount"].mean(skipna=True), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].value_counts().idxmax(), inplace=True)

print(df.isnull().sum())

print('\nDerive new feature:\n')

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
print(df.head())

print('\nSee outliers:\n')

plt.figure(1)
plt.subplot(121)
sns.distplot(df['LoanAmount']);

plt.subplot(122)
df['LoanAmount'].plot.box(figsize=(16,5))
plt.show()

print('\nSQRT transformation and Log transformation to treat outliers:\n')

df['Sqrt_LoanAmount'] = np.sqrt(df['LoanAmount'])
df['Log_LoanAmount'] = np.log(df['LoanAmount'])

print("The skewness of the original data is {}".format(df.LoanAmount.skew()))
print('The skewness of the SQRT transformed data is {}'.format(df.Sqrt_LoanAmount.skew()))
print("The skewness of the LOG transformed data is {}".format(df['Log_LoanAmount'].skew()))

print("The kurtosis of the original data is {}".format(df.LoanAmount.kurt()))
print("The kurtosis of the SQRT transformed data is {}".format(df['Sqrt_LoanAmount'].kurt()))
print("The kurtosis of the LOG transformed data is {}".format(df['Log_LoanAmount'].kurt()))

fig, axes = plt.subplots(1,3,figsize=(15,5))

sns.distplot(df['LoanAmount'], ax=axes[0])
sns.distplot(df['Sqrt_LoanAmount'], ax=axes[1])
sns.distplot(df['Log_LoanAmount'], ax=axes[2])

plt.show()

df1 = df.copy()
df.drop(columns = ['Log_LoanAmount' ,'Sqrt_LoanAmount'], inplace=True)

df1.drop(columns = ['Sqrt_LoanAmount','LoanAmount'], inplace=True)
df1.dtypes

print('\nZ-Score approach to treat Outliers:\n')

df2 = df.copy()
df2['ZR'] = stats.zscore(df2['LoanAmount'])
print(df2.head())

df2= df2[(df2['ZR']>-3) & (df2['ZR']<3)].reset_index()
print(df2.shape, df.shape)

print('\nIQR Method to treat Outliers:\n')

df3 = df.copy()

Q1 = df3.LoanAmount.quantile(0.25)
Q2 = df3.LoanAmount.quantile(0.50)
Q3 = df3.LoanAmount.quantile(0.75)

IQR = Q3 - Q1

LC = Q1 - (1.5*IQR)

UC = Q3 + (1.5*IQR)

print('\nLC\n', LC)
print('\nUC\n', UC)

sns.distplot(df3.LoanAmount)
plt.axvline(UC, color='r')
plt.axvline(LC, color ='r')
plt.axvline(Q1, color='g')
plt.axvline(Q3, color='g')
plt.show()

df3 = df3[(df3.LoanAmount>LC) & (df3.LoanAmount<UC)].reset_index()

print(df3.shape,df.shape)

print('\nScale the dataset with z-score:\n')

df4 = df3.copy()
numeral = ['LoanAmount','ApplicantIncome','CoapplicantIncome']
Z_numeral = ['Z_LoanAmount','Z_ApplicantIncome','Z_CoapplicantIncome']

df4[Z_numeral] = StandardScaler().fit_transform(df4[numeral])
print(df4.head())

fig, axes = plt.subplots(2,2, figsize=(15,8))

sns.distplot(df4['LoanAmount'], ax=axes[0,0])
sns.distplot(df4['Z_LoanAmount'], ax=axes[0,1])
sns.distplot(df4['ApplicantIncome'], ax=axes[1,0])
sns.distplot(df4['Z_ApplicantIncome'], ax=axes[1,1])

plt.show()

print('\nScale the dataset with Min Max Scalar\n')

model = MinMaxScaler()
Min_Max_numeral = ['Min_Max_LoanAmount','Min_Max_ApplicantIncome','Min_Max_CoapplicantIncome']
df4[Min_Max_numeral] = model.fit_transform(df4[numeral])
df4.head()

print('\nEncoding Categorical Features:\n')

df_loans = df3.copy()
print(df_loans.head())

df_loans = pd.get_dummies(df_loans, columns=['Gender','Married','Property_Area'],drop_first=True)
print(df_loans.head())

df_loans['Loan_Status'] = LabelEncoder().fit_transform(df_loans['Loan_Status'])
print(df_loans.head())

df_loans['Dependents'].replace(('0', '1', '2', '3+'),(0, 1, 2, 3),inplace=True)
df_loans['Education'].replace(('Not Graduate', 'Graduate'),(0, 1),inplace=True)
df_loans['Self_Employed'].replace(('No','Yes'),(0,1),inplace=True)
print(df_loans.head())

print('\nPrepare training and testing dataset:\n')

Y = df_loans['Loan_Status']
X = df_loans.drop('Loan_Status', axis=1)

print(X.head())
print(Y.head())

X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.8, random_state =0)

print("The shape of X_train is:", X_train.shape)
print("The shape of X_test is:\n", X_test.shape)

print("The shape of y_train is:", y_train.shape)
print("The shape of y_test is:\n", y_test.shape)

model = LogisticRegression() 
model.fit(X,Y) 

y_prediction = model.predict(X_test) 
print('Logistic Regression accuracy = ', metrics.accuracy_score(y_prediction,y_test))

model = DecisionTreeClassifier()

model.fit(X,Y) 

y_prediction = model.predict(X_test) 

print('Decision Tree accuracy = ', metrics.accuracy_score(y_prediction,y_test))