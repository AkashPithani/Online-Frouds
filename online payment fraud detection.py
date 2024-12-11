#!/usr/bin/env python
# coding: utf-8

# # Online Payment Fraud detection

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('PS_20174392719_1491204439457_log.csv') 


# In[4]:


data.head(6)


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data['type'].value_counts()


# In[8]:


type = data['type'].value_counts()


# In[9]:


transaction = type.index
quantity = type.values


# In[10]:


import plotly.express as px
view = px.pie(data, values = quantity, names = transaction, hole=0.5, title='Distributing of Transaction type')
view.show()


# In[11]:


numerical_data = data.select_dtypes(include=[np.number])


# In[12]:


correlation = numerical_data.corr()


# In[13]:


print(correlation["isFraud"].sort_values(ascending=False))


# In[14]:


data["type"] = data["type"].map({"CASH_OUT": 1,"PAYMENT": 2,"CASH_IN": 3,"TRANSFER": 4,"DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

data.head()


# In[15]:


x = np.array(data[['type','amount','oldbalanceOrg','newbalanceDest']])
y = np.array(data[['isFraud']])


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)
xtrain.shape


# In[17]:


model = DecisionTreeClassifier()
model.fit(xtrain,ytrain)
model.score(xtest,ytest)


# In[18]:


feature = np.array([[4,9000.60,9000.60,0.0]])
model.predict(feature)


# In[ ]:




