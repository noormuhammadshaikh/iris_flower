#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set(style='white',color_codes=True)


# In[2]:


iris =  pd.read_csv('C:/Users/hp/Downloads/flower/IRIS.csv')


# In[3]:


iris.head()


# In[4]:


iris["species"].value_counts()


# In[5]:


sns.FacetGrid(iris, hue="species", height=5).map(plt.scatter, "petal_length", "petal_width").add_legend()


# In[6]:


flower_mapping={'setosa': 0, 'versicolor' : 1, 'virginica' : 2 }
iris["species"] = iris["species"].map(flower_mapping)


# In[7]:


iris.head()


# In[13]:


X=iris[['sepal_length','sepal_width', 'petal_length','petal_width']].values
y=iris[['species']].values


# In[9]:


print(iris.columns)


# In[10]:


from sklearn.linear_model import LogisticRegression


# In[11]:


model = LogisticRegression()


# In[21]:


model.fit(X, y)


# In[15]:


model.score(X, y)


# In[16]:


expected = y
predicted = model.predict(x)
predicted


# In[17]:


from sklearn import metrices


# In[18]:


print(metrics.classification_report(expected,predicted))


# In[19]:


print(metrics.confusion_matrix(expected,predicted))


# In[20]:


model = LogisticRegression(C=20,penalty='12')


# In[ ]:




