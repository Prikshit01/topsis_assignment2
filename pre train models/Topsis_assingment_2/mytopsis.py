#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd


# In[52]:


matrix=pd.read_csv("output.csv")
mat=matrix.iloc[:,1:6]


# In[53]:


mat


# In[54]:


#weights
weights=np.array([1,1,0.5,0.5,1])
impacts=np.array(["+","-","+","-","+"])


# In[55]:


#normalization
normalized_matrix=mat/np.sqrt((mat ** 2).sum(axis=0))
normalized_matrix


# In[58]:


#calculate weighted normalized matrix
weighted_normalized=normalized_matrix*weights


#determine ideal and negative solution
ideal_solution = np.array([
    weighted_normalized.iloc[:, i].max() if impacts[i] == "+" else weighted_normalized.iloc[:, i].min()
    for i in range(normalized_matrix.shape[1])
])

negative_ideal = np.array([
    weighted_normalized.iloc[:, i].min() if impacts[i] == "+" else weighted_normalized.iloc[:, i].max()
    for i in range(normalized_matrix.shape[1])
])


# In[59]:


#eucledian distance

eucledian_dis_ideal=np.sqrt(((weighted_normalized-ideal_solution)**2).sum(axis=1))
eucledian_dis_negative=np.sqrt(((weighted_normalized-negative_ideal)**2).sum(axis=1))


# In[60]:


#performance score
performance=eucledian_dis_negative/(eucledian_dis_ideal+eucledian_dis_negative)


# In[61]:


performance


# In[62]:


#ranking
df=pd.DataFrame(performance)
df['Rank'] = df.rank(ascending=False)


# In[63]:


rank=np.matrix(df.values)


# In[64]:


result=np.concatenate([mat,rank],axis=1)


# In[65]:


df=pd.DataFrame(result)
df


# In[66]:


df.to_csv("result.csv")
