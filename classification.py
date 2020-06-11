#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff # import figure factory

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# close warning
import warnings
warnings.filterwarnings("ignore")

import os

# Any results you write to the current directory are saved as output.


# In[3]:


df = pd.read_csv("/Users/anushainugurthi/Desktop/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


# shape gives number of rows and columns in a tuple
df.shape


# In[7]:


df.describe()


# In[8]:


# Display positive and negative correlation between columns
df.corr()


# In[9]:


#sorts all correlations with ascending sort.
df.corr().unstack().sort_values().drop_duplicates()


# In[10]:


#correlation map
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, linewidth=".5", cmap="RdPu", fmt=".2f", ax = ax)
plt.title("Correlation Map",fontsize=20)
plt.show()


# In[11]:


#if we want use violin plot,we have to convert class variable as integer.
#That's why,we convert object type to int type
#Firstly,we copy the original df
df2 = df.copy()
pd.DataFrame(df2)

df2["class"] = [0 if each == "Abnormal" else 1 for each in df2["class"]]
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=df2, palette=pal, inner="points",x="class",y="pelvic_radius")
plt.show()


# In[12]:


pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=df2, palette=pal, inner="points",x="class",y="sacral_slope")
plt.show()


# In[13]:


pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=df2, palette=pal, inner="points",x="class",y="pelvic_incidence")
plt.show()


# In[14]:


sns.swarmplot(x="class", y="pelvic_radius", data=df)
plt.show()


# In[15]:


sns.swarmplot(x="class", y="sacral_slope", data=df)
plt.show()


# In[16]:


sns.swarmplot(x="class", y="pelvic_incidence", data=df)
plt.show()


# In[17]:


sns.pairplot(data=df,hue="class",palette="Set1")
plt.suptitle("Pair Plot of Data",fontsize=20)
plt.show()   # pairplot without standard deviaton fields of data


# In[18]:


color_list = ["red" if each=="Abnormal" else "cyan" for each in df.loc[:,"class"]]
pd.plotting.scatter_matrix(df.loc[:, df.columns != "class"],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal="hist",
                                       alpha=0.5,
                                       s = 200,
                                       marker = "*",
                                       edgecolor= "black")
plt.show()


# In[19]:


df_abnormal = df[df["class"]=="Abnormal"]
pd.plotting.scatter_matrix(df_abnormal.loc[:, df_abnormal.columns != "class"],
                                       c="red",
                                       figsize= [15,15],
                                       diagonal="hist",
                                       alpha=0.5,
                                       s = 200,
                                       marker = "*",
                                       edgecolor= "black")
plt.show()


# In[20]:


df_normal = df[df['class']=='Normal']
pd.plotting.scatter_matrix(df_normal.loc[:, df_normal.columns != "class"],
                                       c="cyan",
                                       figsize= [15,15],
                                       diagonal="hist",
                                       alpha=0.5,
                                       s = 200,
                                       marker = "*",
                                       edgecolor= "black")
plt.show()


# In[36]:


df["class"] = [0 if each == "Abnormal" else 1 for each in df["class"]]

y = df["class"].values
x_data = df.drop(["class"], axis=1)


# In[37]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[38]:


x.head()


# In[39]:


x.isnull().sum() #Indicates values not defined in our data


# In[40]:


x.isnull().sum().sum()  #Indicates sum of values in our data


# In[41]:


print(x.shape)
print(y.shape)


# In[42]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# In[43]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)

#Print Train Accuracy
lr_train_accuracy = lr_model.score(x_train,y_train)
print("lr_train_accuracy = ",lr_model.score(x_train,y_train))
#Print Test Accuracy
lr_test_accuracy = lr_model.score(x_test,y_test)
print("lr_test_accuracy = ",lr_model.score(x_test,y_test))


# In[56]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train,y_train)

#Print Train Accuracy
knn_train_accuracy = knn_model.score(x_train,y_train)
print("knn_train_accuracy = ",knn_model.score(x_train,y_train))
#Print Test Accuracy
knn_test_accuracy = knn_model.score(x_test,y_test)
print("knn_test_accuracy = ",knn_model.score(x_test,y_test))


# In[ ]:





# In[58]:


# Model complexity
neighboors = np.arange(1,30)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neighboors):
    # k from 1 to 30(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # fit with knn
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))           # train accuracy
    test_accuracy.append(knn.score(x_test, y_test))              # test accuracy



knn_train_accuracy = np.max(train_accuracy)
knn_test_accuracy = np.max(test_accuracy)
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))


# In[62]:


from sklearn.svm import SVC

svm_model = SVC(random_state=1)
svm_model.fit(x_train,y_train)

#Print Train Accuracy
svm_train_accuracy = svm_model.score(x_train,y_train)
print("svm_train_accuracy = ",svm_model.score(x_train,y_train))
#Print Test Accuracy
svm_test_accuracy = svm_model.score(x_test,y_test)
print("svmr_test_accuracy = ",svm_model.score(x_test,y_test))


# In[63]:


from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(x_train,y_train)

#Print Train Accuracy
nb_train_accuracy = nb_model.score(x_train,y_train)
print("nb_train_accuracy = ",nb_model.score(x_train,y_train))
#Print Test Accuracy
nb_test_accuracy = nb_model.score(x_test,y_test)
print("nb_test_accuracy = ",nb_model.score(x_test,y_test))


# In[64]:


from sklearn.tree import DecisionTreeClassifier
#if you remove random_state=1, you can see how accuracy is changing
#Accuracy changing depends on splits
dt_model = DecisionTreeClassifier(random_state=1)
dt_model.fit(x_train,y_train)

#Print Train Accuracy
dt_train_accuracy = dt_model.score(x_train,y_train)
print("dt_train_accuracy = ",dt_model.score(x_train,y_train))
#Print Test Accuracy
dt_test_accuracy = dt_model.score(x_test,y_test)
print("dt_test_accuracy = ",dt_model.score(x_test,y_test))


# In[65]:


from sklearn.ensemble import RandomForestClassifier

#n_estimators = 100 => Indicates how many trees we have
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
rf_model.fit(x_train,y_train)

#Print Train Accuracy
rf_train_accuracy = rf_model.score(x_train,y_train)
print("rf_train_accuracy = ",rf_model.score(x_train,y_train))
#Print Test Accuracy
rf_test_accuracy = rf_model.score(x_test,y_test)
print("rf_test_accuracy = ",rf_model.score(x_test,y_test))


# In[67]:


y_pred = lr_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_true,y_pred)


# In[68]:


y_pred = knn_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_knn = confusion_matrix(y_true,y_pred)


# In[69]:


y_pred = svm_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_svm = confusion_matrix(y_true,y_pred)


# In[70]:


y_pred = nb_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_nb = confusion_matrix(y_true,y_pred)


# In[71]:


y_pred = dt_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_dt = confusion_matrix(y_true,y_pred)


# In[72]:


y_pred = rf_model.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_true,y_pred)


# In[84]:


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


plt.figure(figsize=(20,10))
plt.suptitle("Confusion Matrixes of Classification Models",fontsize=30)


plt.subplot(2,3,1)
plt.title("Logistic Regression Classification")
sns.heatmap(cm_lr,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,2)
plt.title("Decision Tree Classification")
sns.heatmap(cm_knn,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,3)
plt.title("K Nearest Neighbors(KNN) Classification")
sns.heatmap(cm_svm,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,4)
plt.title("Naive Bayes Classification")
sns.heatmap(cm_nb,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,5)
plt.title("Random Forest Classification")
sns.heatmap(cm_dt,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)

plt.subplot(2,3,6)
plt.title("Support Vector Machine(SVM) Classification")
sns.heatmap(cm_rf,annot=True,cmap='YlGnBu',fmt=".0f",cbar=False)



plt.show()


# In[102]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('L R', 'KNN', 'SVM', 'N B', 'D T', 'R F')

performance = [0.7526881720430108,0.8172043010752689,0.6881720430107527,0.8172043010752689,0.7849462365591398,0.8602150537634409]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)



plt.title('test-accuracy')

plt.show()


# In[103]:


objects = ('L R', 'KNN', 'SVM', 'N B', 'D T', 'R F')

performance = [0.7649769585253456,0.8172043010752689,0.663594470046083,0.7695852534562212,1.0, 0.9953917050691244]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)



plt.title('train-accuracy')

plt.show()


# In[ ]:




