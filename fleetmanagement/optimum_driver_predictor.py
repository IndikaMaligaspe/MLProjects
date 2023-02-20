import pandas as pd 
import numpy as np 
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from datetime import datetime as dt


# load dataset
df = pd.read_csv('G:/workspace/IM.tech.solutions/DK_Care/MAS/Documentation/ML/analysis_2.csv')

#check for any missing columns
print(df.isna().sum())

#print stats
# print(df.describe())


# Transform pickup / dropoff dates to year / month / date
df['pickup_date'] = pd.to_datetime(df['pickupdate'])
df['pk_year'] = df['pickup_date'].dt.year
df['pk_month'] = df['pickup_date'].dt.month
df['pk_date'] = df['pickup_date'].dt.day
df['drop_time'] = df['dropofftime'].replace(to_replace ="call", value="0000")
df['dropoff_date'] = pd.to_datetime(df['dropoffdate'])
df['dp_year'] = df['dropoff_date'].dt.year
df['dp_month'] = df['dropoff_date'].dt.month
df['dp_date'] = df['dropoff_date'].dt.day


clensed_df = df.drop(['dropoff_date','name','vehicleId','pickupdate', 'pickup_date','dropoffdate','dropofftime'], axis='columns')

# ohe = OneHotEncoder()
# ct = make_column_transformer(
#     (ohe, ["medicaidNumber"]),
#     remainder="drop"
# )


X = clensed_df.drop(columns=['driverId'])
y = clensed_df['driverId']

print(X.columns.values)
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)



model = DecisionTreeClassifier()
# model = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(predictions, y_test)
print(score)
# print(predictions)

m_test = pd.DataFrame([{"invoiceNumber":"1234475830",
                        "medicaidNumber":"56",
                        "name":"mariasarine",
                        "pickuplat":"41.739217",
                        "pickuplng":"41.739217",
                        "droplat":"41.3044629",
                        "sroplng":"41.3044629",
                        "pickupdate":"2/26/2023",
                        "dropoffdate":"2/26/2023",
                        "pickuptime":"1000",
                        "dropoffdate":"2/26/2023",
                        "standingOrder":"1",
                        "dropofftime":"1130"
                    }])	

m_test['pickup_date'] = pd.to_datetime(m_test['pickupdate'])
m_test['pk_year'] = m_test['pickup_date'].dt.year
m_test['pk_month'] = m_test['pickup_date'].dt.month
m_test['pk_date'] = m_test['pickup_date'].dt.day
m_test['drop_time'] = m_test['dropofftime'].replace(to_replace ="call", value="0000")
m_test['dropoff_date'] = pd.to_datetime(m_test['dropoffdate'])
m_test['dp_year'] = m_test['dropoff_date'].dt.year
m_test['dp_month'] = m_test['dropoff_date'].dt.month
m_test['dp_date'] = m_test['dropoff_date'].dt.day

m_clensed_df = m_test.drop(['dropoff_date','name','pickupdate', 'pickup_date','dropoffdate','dropofftime'], axis='columns')
print(m_clensed_df.columns.values)
# m_fit_data = ct.fit_transform(m_clensed_df)

m_prediction = model.predict(m_clensed_df)
print(m_prediction)