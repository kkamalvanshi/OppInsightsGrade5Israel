#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import RidgeCV, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold


# In[2]:


df = pd.io.stata.read_stata('C://Users//kkama//Downloads//project2//grade5.dta')


# In[3]:


df.to_csv('grade5-3.csv')
df


# In[4]:


df = df.drop(['schlcode', 'avgmath', 'grade'], axis = 'columns')


# In[5]:


df["disadvantaged"] = 0.01 * df["disadvantaged"]
df["disadvantaged"]


# In[6]:


df


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df.hist(bins=20, figsize=(20,20))
plt.show()


# In[10]:


df


# In[11]:


df_copy = df.copy();


# In[12]:


df_train = df_copy.sample(frac=0.80, random_state=0)


# In[13]:


df_test = df_copy.drop(df_train.index)


# In[14]:


df_train


# In[15]:


df_train.hist(bins=20, figsize=(20,20))
plt.show()


# In[16]:


df_test


# In[17]:


from sklearn.linear_model import LinearRegression
model_verb = LinearRegression(normalize=True)


# In[18]:


model_verb_rfr = RandomForestRegressor(n_estimators = 100)


# In[19]:


X_verb=df_train.drop(['avgverb'],axis='columns')
X_verb


# In[20]:


Y_verb=df_train.avgverb
Y_verb


# In[21]:


model_verb_rfr.fit(X_verb,Y_verb)


# In[22]:


score_rfr = model_verb_rfr.score(X_verb, Y_verb)
score_rfr


# In[23]:


df_test


# In[26]:


df_test_avgverb=df_test.avgverb


# In[27]:


df_test_avgverb


# In[28]:


df_final_test=df_test.drop(['avgverb'], axis = 'columns')


# In[29]:


df_final_test


# In[30]:


X_test_verb = df_final_test
X_test_verb


# In[31]:


Y_test_verb = model_verb_rfr.predict(X_test_verb)


# In[32]:


Y_test_verb


# In[35]:


df_test_avgverb_array = df_test_avgverb.to_numpy(dtype=None, copy=False)
df_test_avgverb_array


# In[36]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(df_test_avgverb_array, Y_test_verb))
rms


# In[38]:


df_Y_test_verb= pd.DataFrame(data = Y_test_verb, columns = ['avgverb'])
df_Y_test_verb


# In[39]:


plt.scatter(df_test_avgverb_array,Y_test_verb)   
plt.plot(df_test_avgverb_array, df_test_avgverb_array, color = 'red') 
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')


# HOW TO INPUT: school_enrollment, classize, disadvantaged, female, religious

# In[41]:


model_verb_rfr.predict([[95, 32, 0.06, 0.473684, 0.0]])


# In[ ]:




