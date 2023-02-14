import pandas as pd 
import numpy as np 
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#load dataset
df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.shape)
print(df.columns.values)

#check for any missing columns
print(df.isna().sum())

#get some stats
print(df.describe())

#customer churn count
print(df['Churn'].value_counts())

# clean data - drop unwanted columns
telco_churn_df = df.drop('customerID', axis=1)

#visualise the churn
sns.countplot(telco_churn_df['Churn'])
plt.show()
# This shows more customers have stayed with the company then left

#%customer left or stayed
num_stayed = telco_churn_df[telco_churn_df['Churn'] == 'No'].shape[0]
num_left = telco_churn_df[telco_churn_df['Churn'] == 'Yes'].shape[0]

print(num_stayed / (num_stayed+ num_left) * 100, '% customers stayed.')
print(num_left / (num_stayed + num_left) * 100, '% customers left.')

#visualise the curn count between male vs female
sns.countplot(x='gender', hue = 'Churn', data=telco_churn_df )
plt.show()
# this shows that gender has no major impact to churn

#visualise the curn count for internet servive
sns.countplot(x='InternetService', hue = 'Churn', data=telco_churn_df )
plt.show()
# Very Interesting : this shows most of the customers that stayed had DSL and most customers left have Fibre Optices

#Do a histogram of tenure , Monthy_charges
num_props = ['tenure','MonthlyCharges']
fig, ax = plt.subplots(1, 2, figsize=(28,8))
telco_churn_df[telco_churn_df['Churn'] == 'No'][num_props].hist(bins=20, color='blue', alpha=0.5, ax = ax)
telco_churn_df[telco_churn_df['Churn'] == 'Yes'][num_props].hist(bins=20, color='orange', alpha=0.5, ax = ax)
plt.show()
# Very Interesting : this shows most of the customers that stayed has longer tenure DSL and most customers left have have extreamly short tenue.

#clean data - convert all non numerics to numbers
for column in telco_churn_df.columns:
    if telco_churn_df[column].dtype == np.number:
        continue
    else:
        telco_churn_df[column] = LabelEncoder().fit_transform(telco_churn_df[column])

#Sclate the dataset
x = telco_churn_df.drop('Churn', axis=1) # Fetaure data set
y = telco_churn_df['Churn'] # Target data set

x = StandardScaler().fit_transform(x)

#split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create the model
model =  LogisticRegression()
model.fit(x_train, y_train)

#predictions
predictions = model.predict(x_test)
print(predictions)

# Check the precission
print(classification_report(y_test, predictions))

