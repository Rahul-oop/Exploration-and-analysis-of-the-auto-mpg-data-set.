#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[16]:


import warnings 
warnings.filterwarnings('ignore')


# In[17]:


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# In[18]:


df=pd.read_csv("auto-mpg.csv")


# In[19]:


df.shape


# In[20]:


df.dtypes


# In[21]:


df.isnull().sum()


# In[22]:


df.head()


# In[23]:


df["hp"].value_counts()


# In[24]:


# replace question marks
df["hp"]=df["hp"].replace("?",np.nan)
df["hp"]=df["hp"].astype(float)


# In[25]:


df.dtypes


# In[26]:


# if we wants to find statistical measure 
df.describe()


# In[1]:


df["hp"]=df["hp"].replace(np.nan,df["hp"].mean())


# In[28]:


df.describe()


# In[30]:


# replace origin value 
df["origin"]=df["origin"].replace({1:"america",2:"India",3:"Europe"})


# In[31]:


df.sample(10)


# In[32]:


df.hist(figsize=(10,12))
plt.show()


# In[33]:


# measure skewness of this graph
df.skew(numeric_only=True)


# In[35]:


sns.boxplot(x="mpg",data=df)


# In[36]:


sns.boxplot(x="cyl",data=df)


# In[37]:


# visulalizing data with the help of histogram
import plotly.express as px
for column in df:
    fig=px.histogram(df, x=column, nbins=20 )
    fig.show()


# In[38]:


for column in df:
    fig=px.box(df, x=column)
    fig.show()


# In[39]:


plt.scatter(df["hp"],df["mpg"])


# In[40]:


plt.scatter(df["wt"],df["acc"])


# In[41]:


sns.countplot(x='origin',data=df)


# In[42]:


sns.violinplot(x='origin',y='mpg',data=df)


# In[43]:


sns.barplot(x='cyl',y='mpg',data=df)


# In[44]:


sns.lineplot(x='yr',y='mpg',data=df)


# In[45]:


sns.jointplot(x='wt',y='mpg',data=df)


# In[46]:


# to create a corelattion
corr_matrix=df.corr(numeric_only=True)
corr_matrix


# In[47]:


sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


# In[48]:


sns.kdeplot(x='wt',y='mpg',data=df)


# In[49]:


sns.boxenplot(x='wt',y='mpg',data=df)


# In[ ]:




