
# coding: utf-8

# In[1]:


import sqlite3
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
from dateutil import relativedelta
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score


# In[2]:


cnx = sqlite3.connect('database.sqlite')
tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", cnx)
tables


# In[3]:


PA = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
print(PA.shape)
PA.head()


# In[4]:


PA.describe()


# In[5]:


Player = pd.read_sql_query("SELECT * FROM Player", cnx)
print(Player.shape)
Player.head()


# In[6]:


Player.describe()


# In[7]:


DS = pd.read_sql_query("SELECT * FROM Player_Attributes PA INNER JOIN Player P ON PA.player_api_id = P.player_api_id", cnx)
DS.head()


# In[8]:


DS['age'] = 1

def age(dob):
    dobasdt = datetime.strptime(dob, '%Y-%m-%d %H:%M:%S')
    today = datetime.now()
    difference = relativedelta.relativedelta(today, dobasdt)
    
    return difference.years

DS['age'] = DS.birthday.apply(age).astype('float64', copy=False)
DS = DS.dropna()
Y = DS.overall_rating


# In[9]:


DS.columns


# In[10]:


DS.drop(['id', 'player_fifa_api_id', 'player_api_id', 'date', 'player_name', 'birthday', 'overall_rating'], axis = 1, inplace=True)


# In[11]:


DS.head()


# In[12]:


dummy_PF = pd.get_dummies(DS['preferred_foot'], prefix='preferred_foot')
dummy_PF = dummy_PF.astype('float64', copy=False)
dummy_PF.head()


# In[13]:


data = DS.join(dummy_PF.ix[:, 'preferred_foot_left':])
del data['preferred_foot']
DS = data
DS.head()


# In[14]:


dummy_AWR = pd.get_dummies(DS['attacking_work_rate'], prefix='attacking_work_rate')
dummy_AWR = dummy_AWR.astype('float64', copy=False)
dummy_AWR.head()


# In[15]:


data = DS.join(dummy_AWR.ix[:, 'attacking_work_rate_None':])
del data['attacking_work_rate']
DS = data
DS.head()


# In[16]:


dummy_DWR = pd.get_dummies(DS['defensive_work_rate'], prefix='defensive_work_rate')
dummy_DWR = dummy_DWR.astype('float64', copy=False)
dummy_DWR.head()


# In[17]:


data = DS.join(dummy_DWR.ix[:, :])
del data['defensive_work_rate']
DS = data
DS.head()


# In[18]:


lm = LinearRegression()
lm.fit(DS, Y)

print(lm.intercept_)
print(lm.coef_)


# In[19]:


y_pred = lm.predict(DS)
print(mean_squared_error(Y, y_pred))


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(DS, Y, test_size=0.25, random_state=123456)

lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.intercept_)
print(lm.coef_)

print("*" * 50)

y_pred = lm.predict(X_test)
print(mean_squared_error(y_test, y_pred))


# In[21]:


scores = cross_val_score(LinearRegression(), DS, Y, scoring='neg_mean_squared_error', cv=10)
scores, scores.mean()

