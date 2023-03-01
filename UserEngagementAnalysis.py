#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import shap

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


users = pd.read_csv('', 
                   encoding = "ISO-8859-1")
print(f'There are {users.duplicated().sum()} duplicated entries.')
print(users.shape)
users.head()


# In[3]:


users.describe()


# In[4]:


users.dtypes


# In[5]:


users['last_session_creation_time'] = pd.to_datetime(users['last_session_creation_time'],unit='s')


# In[6]:


users['creation_time'] = pd.to_datetime(users['creation_time'])


# In[7]:


users['creation_source'].value_counts().plot(kind='bar')
plt.title('Creation Source Counts')
plt.show()


# In[8]:


print(f'The earliest account creation is: {users["creation_time"].min()}')
print(f'The most recent account creation is:{users["creation_time"].max()}\n')

print(f'The earliest date for last login is: {users["last_session_creation_time"].min()}')
print(f'The most recent date for last login is: {users["last_session_creation_time"].max()}')


# In[9]:


engagement = pd.read_csv('', 
                   encoding = "ISO-8859-1")
print(f'There are {engagement.duplicated().sum()} duplicated entries.')
print(engagement.shape)
engagement.head()


# In[10]:


#Creating filter to get user ids with three or more logins

login_filter = engagement.groupby('user_id')['user_id'].filter(lambda x: len(x) >= 3)
adopted_engagement = engagement[engagement['user_id'].isin(login_filter)]


# In[11]:


#Creating a new column that contains the date for one week from each login timestamp

week = []
adopted_engagement['time_stamp'] = pd.to_datetime(adopted_engagement['time_stamp'])
for timestamp in adopted_engagement['time_stamp']:
    week.append(timestamp + datetime.timedelta(days=7))
    
adopted_engagement['week'] = week
adopted_engagement.head()


# In[12]:


#Resetting the index for the new engagement dataframe 

adopted_engagement = adopted_engagement.reset_index()
adopted_engagement = adopted_engagement.drop(columns=['index'])
adopted_engagement.head()


# In[13]:


#Creating a new column for adopted users
#The loop checks that the next two timestamps for one user's login timestamp are within one week of each other

adopted_user = []

for index in range(0,201000):
    if adopted_engagement['user_id'][index] == adopted_engagement['user_id'][index+1]:
        if (adopted_engagement['time_stamp'][index+1] < adopted_engagement['week'][index]) & (adopted_engagement['time_stamp'][index+2] < adopted_engagement['week'][index]):
            adopted_user.append('yes')
        else:
            adopted_user.append('no')
    else:
        adopted_user.append('diff_user')
        
adopted_user.extend(['outside_idx','outside_idx'])
adopted_engagement['adopted_user'] = adopted_user 
adopted_engagement.head()


# In[14]:


#Creating an adopted users column for the users dataframe

confirmed = adopted_engagement.query('adopted_user == "yes"')

adopted = list(confirmed['user_id'])
confirmed_adopted = []

for user in users['object_id']:
    if user in adopted:
        confirmed_adopted.append('1')
    else:
        confirmed_adopted.append('0')
        
users['confirmed_adopted'] = confirmed_adopted
users.head()


# In[15]:


#FEATURE ENGINEERING
#Making a new column for if a user was invited

invited = []
all_users = list(users['object_id'])

for user in users['invited_by_user_id']:
    if user in all_users:
        invited.append(1)
    else:
        invited.append(0)
        
users['invited'] = invited
users.head()


# In[16]:


#FEATURE ENGINEERING
#Making a new column for if a user invited someone else

inviter = []
inviters = list(users['object_id'])

for user in users['invited_by_user_id']:
    if user in inviters:
        inviter.append(1)
    else:
        inviter.append(0)

users['inviter'] = inviter
users.head()


# In[17]:


#FEATURE ENGINEERING
#Making a new column for the current age of a user's account, or the time between account creation and last login

account_age = []

for index in users.index:
    try:
        account_age.append((users['last_session_creation_time'][index] - users['creation_time'][index]).days)
    except:
        account_age.append(0)
users['account_age'] = account_age
users.head()


# In[18]:


users['account_age'] = users['account_age'].fillna(0)
users.isnull().sum()


# In[19]:


#Preparing features for modeling

variables = pd.get_dummies(users['creation_source'])
users2 = users.drop(['creation_source','invited_by_user_id', 'last_session_creation_time', 
                         'email', 'name', 'object_id', 'creation_time'],axis=1)
df = pd.concat([users2,variables], axis=1)
df.head()


# In[20]:


x = df.drop(['confirmed_adopted'],axis=1)
y = df['confirmed_adopted']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, random_state=1)


# In[21]:


#Testing four tree-based models for their average accuracy and balanced accuracy

models = []

models.append(('CART', DecisionTreeClassifier()))
models.append(('XGB', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('BRF', BalancedRandomForestClassifier()))

accuracy = []
balanced_accuracy = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_validate(model, x_train, y_train, cv=kfold, 
                            scoring=['accuracy','balanced_accuracy'],return_train_score=True)
    accuracy.append(scores['test_accuracy'])
    balanced_accuracy.append(scores['test_balanced_accuracy'])
    names.append(name)
    msg = "%s: Average Accuracy (%f); Average Balanced Accuracy (%f)" % (name, scores['test_accuracy'].mean(),scores['test_balanced_accuracy'].mean())
    print(msg)


# In[22]:


brf = BalancedRandomForestClassifier(random_state=1)

brf.fit(x_train,y_train)
y_pred = brf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=brf.classes_)
disp.plot()

plt.show()


# In[23]:


print(brf.feature_importances_)
plt.barh(x_test.columns, brf.feature_importances_)


# In[24]:


#Generating shap values for the model

explainer = shap.explainers.Tree(brf)
shap_values = explainer.shap_values(x_test)
shap_test = explainer(x_test)


# In[25]:


shap.plots.beeswarm(shap_test[:,:,1])


# In[26]:


#Separating individual predictions based on if they were correct or not 

test_df = pd.concat([x_test,y_test],axis=1)
test_df['probability_adopting'] = brf.predict_proba(x_test)[:,1]
test_df['test_set_idx'] = np.arange(len(test_df))

false_negatives = test_df.query('confirmed_adopted == "1" & probability_adopting < 0.50')
true_negatives = test_df.query('confirmed_adopted == "0" & probability_adopting < 0.50')
false_positives = test_df.query('confirmed_adopted == "0" & probability_adopting > 0.50')
true_positives = test_df.query('confirmed_adopted == "1" & probability_adopting > 0.50')


# In[27]:


def shap_explanation(df, index,label=1):
    """This function creates a waterfall plot for a given row in a specified dataframe."""
    idx = int(df.iloc[index]['test_set_idx'])
    shap.plots.waterfall(shap_test[:,:,label][idx],max_display=11)


# In[28]:


shap_explanation(df=true_positives,index=157)


# In[29]:


shap_explanation(df=true_positives,index=15)


# In[31]:


shap_explanation(df=false_negatives,index=5)


# In[35]:


shap_explanation(df=false_negatives,index=3)

