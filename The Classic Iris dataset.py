#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


# In[2]:


sns.set()


# LOAD DATA

# In[3]:


data =datasets.load_iris()


# In[4]:


data.keys()


# In[5]:


print(data["DESCR"])


# In[6]:


data['data']


# In[7]:


data['data'][:5]


# In[8]:


data['feature_names']


# In[9]:


data['target']


# In[10]:


data['target_names']


# # what are trying to do
# we are trying to use attribute of flower to predict the flower species

# In[11]:


df=pd.DataFrame(data['data'],columns=data['feature_names'])


# In[12]:


df['target']=data['target']


# In[13]:


df.head()


# In[14]:


df.describe()


# In[15]:


col='sepal length (cm)'
df[col].hist()
plt.suptitle(col)
plt.show()


# In[16]:


col='sepal width (cm)'
df[col].hist()
plt.suptitle(col)
plt.show()


# In[17]:


col='petal width (cm)'
df[col].hist()
plt.suptitle(col)
plt.show()


# In[18]:


col='petal length (cm)'
df[col].hist()
plt.suptitle(col)
plt.show()


# In[19]:


data['target_names']


# In[20]:


#Create new column with the species name
df['target_names']=df['target'].map({0:'setosa',1:'versicolor',2:'virginica'})


# In[21]:


df


# In[22]:


col='sepal length (cm)'
sns.relplot(x=col,y='target',hue='target_names',data=df)
_=plt.suptitle(col, y=1.05)


# In[23]:


col='sepal width (cm)'
sns.relplot(x=col,y='target',hue='target_names',data=df)
_=plt.suptitle(col, y=1.05)


# In[24]:


col='petal width (cm)'
sns.relplot(x=col,y='target',hue='target_names',data=df)
_=plt.suptitle(col, y=1.05)


# In[25]:


col='petal length (cm)'
sns.relplot(x=col,y='target',hue='target_names',data=df)
_=plt.suptitle(col, y=1.05)


# In[26]:


sns.pairplot(df,hue='target_names')


# #Train test split

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


df_train,df_test = train_test_split(df,test_size=0.25)


# In[29]:


df_train.shape


# In[30]:


df_test.shape


# In[31]:


df_train.head()


# #prepare our data for modeling
# this involves splitting the data back out into plain numpy array

# In[32]:


x_train=df_train.drop(columns=['target','target_names']).values
y_train=df_train["target"].values


# In[33]:


y_train


# # Modeling -What is our baseline?
# what is the simplest model we can think off?
# 
# In this case,if our baseline model is just randomly guessing the flower, or guessing a species for every data point, we would expect to have a model accurancy of 0.33 or 33%,since we have 3 different classes that are evenly balanced 

# # Modeling - simple manual model
# let's manually look at our data and decide

# In[34]:


def single_feature_prediction(petal_length):
    "Predicts the iris species given the petal length"
    if petal_length < 2.5:
        return 0
    elif petal_length < 4.8:
        return 1
    else:
        return 2


# In[35]:


x_train[:,2]


# In[36]:


manual_y_prediction=np.array([single_feature_prediction(val) for val in x_train[:,2]])


# In[37]:


manual_model_accuracy =np.mean(y_train == manual_y_prediction)


# In[38]:


print(f"Manual Model Accuracy: {manual_model_accuracy *100:.2f}%")


# # Model Logistic Regression

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


model=LogisticRegression(max_iter=200)


# In[41]:


xt,xv,yt,yv = train_test_split(x_train,y_train, test_size=0.25)


# In[42]:


model.fit(xt,yt)


# In[43]:


model.score(x_train,y_train)   #you never want to evalute your model on the same data that was used for training


# In[44]:


model.fit(xt,yt)


# In[45]:


y_pred=model.predict(xv)


# In[46]:


np.mean(y_pred==yv)


# In[47]:


model.score(xv,yv)


# # Using cross validation  to evaluation our model

# In[48]:


from sklearn.model_selection import cross_val_score,cross_val_predict


# In[49]:


model =LogisticRegression(max_iter=200)


# In[50]:


accuracies=cross_val_score(model,x_train,y_train,cv=5,scoring="accuracy")


# In[51]:


np.mean(accuracies)


# # Where are we misclassiyoing points?

# In[52]:


y_pred=cross_val_predict(model,x_train,y_train,cv=5)


# In[53]:


prediction_correctly_mask= y_pred==y_train


# In[54]:


not_prediction_correctly= ~prediction_correctly_mask


# In[55]:


x_train[not_prediction_correctly]


# In[56]:


df_prediction=df_train.copy()


# In[57]:


df_prediction["correct_prediction"]=prediction_correctly_mask


# In[58]:


df_prediction["prediction"]=y_pred


# In[59]:


df_prediction["prediction_label"]=df_prediction["prediction"].map({0:'setosa',1:'versicolor',2:'virginica'})


# In[60]:


df_prediction.head()


# In[61]:


sns.scatterplot(x="petal length (cm)",y="petal width (cm)",hue="prediction_label",data=df_prediction)


# In[62]:


sns.scatterplot(x="petal length (cm)",y="petal width (cm)",hue="target_names",data=df_prediction)


# In[63]:


def plot_incorrect_prediction(df_prediction,x_axis_feature,y_axis_feature):
    fig, axs = plt.subplots(2,2,figsize=(10,10))
    axs=axs.flatten()
    sns.scatterplot(x=x_axis_feature,y=y_axis_feature,hue="prediction_label",data=df_prediction,ax=axs[0])
    sns.scatterplot(x=x_axis_feature,y=y_axis_feature,hue="target_names",data=df_prediction,ax=axs[1])
    sns.scatterplot(x=x_axis_feature,y=y_axis_feature,hue="correct_prediction",data=df_prediction,ax=axs[2])
    axs[3].set_visible(False)
    plt.show()


# In[64]:


plot_incorrect_prediction(df_prediction,"petal length (cm)","petal width (cm)")


# # Model tuning

# In[65]:


from sklearn.ensemble import RandomForestClassifier


# In[66]:


model = RandomForestClassifier()


# In[67]:


accs = cross_val_score(model,x_train,y_train,cv=5,scoring="accuracy")


# In[68]:


np.mean(accs)


# In[69]:


for reg_param in(1,1.3,2,5,10,100):
    print(reg_param)
    model =LogisticRegression(max_iter=200,C=1)
    accss=cross_val_score(model,x_train,y_train,cv=5,scoring="accuracy")
    print(f"Accurancy: {np.mean(accss)*100:.2f}%")


# # Final Model
# 

# In[70]:


model=LogisticRegression(max_iter=200,C=2)


# In[71]:


x_test = df_test.drop(columns=["target","target_names"]).values
y_test = df_test["target"].values


# In[72]:


x_test.shape


# In[73]:


y_test


# In[74]:


model.fit(x_train,y_train)


# In[75]:


y_test_pred=model.predict(x_test)


# In[76]:


test_set_correctly_classified = y_test_pred == y_test
test_set_accuracy = np.mean(test_set_correctly_classified)


# In[77]:


print(f"Test set accuracy: {test_set_accuracy*100:.2f}")


# In[78]:


test_set_correctly_classified


# In[79]:


df_prediction_test=df_test.copy()
df_prediction_test["correct_prediction"]=test_set_correctly_classified
df_prediction_test["prediction"]=y_test_pred
df_prediction_test["prediction_label"]=df_prediction_test["prediction"].map({0:'setosa',1:'versicolor',2:'virginica'})


# In[80]:


df_prediction_test.head()


# In[81]:


plot_incorrect_prediction(df_prediction_test,x_axis_feature="petal length (cm)",y_axis_feature="petal width (cm)")


# # In Conclusion...
# In conclusion,we achieved a 97% accuracy on the test dataset using a Logisyic Regression model with these model parameters:
# 
# LogisticRegression(C=2, max_iter=200)
# **congratulations on finishing a data science classification mini-project on the Iris dataset!
# 
# 
