#!/usr/bin/env python
# coding: utf-8

# # AI Trading

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import os


# In[2]:


df = pd.read_csv('/Users/armin/Desktop/Gold.csv')
df = df.set_index(pd.DatetimeIndex(df['Date/Time'].values))
df.index.name = 'Date'


# In[3]:


df['Price_Up'] = np.where(df['Close'].shift(-1) > df['Close'],1,0)

df = df.drop(columns= ['Date/Time'])
df = df.drop(columns= ['Name'])
df = df.drop(columns= ['Ticker'])


# In[4]:


df


# ## AI Part

# ### Creating and Training

# Training -> 80% Testing -> 20%

# In[6]:


X = df.iloc[:, 0:df.shape[1]-1].values
Y = df.iloc[:, df.shape[1]-1].values


# In[7]:


X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)
tree = DecisionTreeClassifier().fit(X_train, Y_train)


# ### Testing

# In[8]:


print(tree.score(X_test, Y_test))


# In[ ]:




