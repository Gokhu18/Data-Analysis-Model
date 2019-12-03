#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl
import calendar


# In[35]:


df_features=pd.read_csv(r'C:\Jasbir\Dataset Kaggle\Features_data.csv')
df_sales=pd.read_csv(r'C:\Jasbir\Dataset Kaggle\sales_data.csv')
df_stores=pd.read_csv(r'C:\Jasbir\Dataset Kaggle\stores_data.csv')


# In[20]:


df.head()


# In[21]:


df.info()


# In[7]:


df.isna()


# In[8]:


df.describe()


# In[23]:


print(df_features.columns.tolist())


# In[24]:


print(df_sales.columns.tolist())


# In[25]:


print(df_stores.columns.tolist())


# In[30]:


df_features.astype


# In[62]:


df_features['Date']=pd.to_datetime(df_features['Date'])
df_sales['Date']=pd.to_datetime(df_sales['Date'])


# # Weekly sales for each stores 

# In[36]:


fig, axarr = pl.subplots(7,7,sharex=True, sharey=True,figsize=(15,10))
#fig is important if you want to change the figure level attribute or want to save image file later
# sharex and sharey: control sharing of properties among x (sharex) and y (sharey) axes.
s=1
for i in range(0,7):
    for j in range(0,7):
        xxx = axarr[i,j].hist(df_sales['Weekly_Sales'].loc[df_sales['Store']== s],50);
        axarr[i,j].set_yscale('log') #matplotlib.axes.Axes.set_yscale, Axes.set_yscale(self, value, **kwargs)
        axarr[i,j].set_xscale('log') #value : {"linear", "log", "symlog", "logit", ...} 
        axarr[i,j].set_ylim(1,1e4) #Axes.set_ylim(self, bottom=None, top=None, emit=True, auto=False, *,
                                   #ymin=None, ymax=None)
        axarr[i,j].set_xlim(5e2,1e6) # set the y axis or x axis view limit
        
        s+=1
fig.text(0.5, 0.04, 'Weekely Sales', ha= 'center')
fig.text(0.04,0.5, 'Number', va='center', rotation='vertical')


# In[43]:


print(df_sales[['Store']])


# In[45]:


print(df_sales[['Weekly_Sales']])


# # Weekly sales for week including holiday or not

# In[53]:


fig, axarr = pl.subplots(7,7, sharex=True, sharey=True,figsize=(15,10))
s=1
for i in range(0,7):
    for j in range(0,7):
        xxx= axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['Store']== s) & (df_sales['IsHoliday']== False)],20,
                             color='b',density=True,histtype='step')
        xxx= axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['Store']== s)& (df_sales['IsHoliday']== True)],20,
                           color='r',density=True,histtype='step')
        axarr[i,j].set_yscale('log')
        axarr[i,j].set_xscale('log')
        axarr[i,j].set_ylim(1e-10,5e-4)
        axarr[i,j].set_xlim(5e2,1e6)
        s+=1
fig.text(0.5,0.04, 'Weekly_Sales', ha='center')
fig.text(0.04,0.5,'Normalized umber', va='center',rotational='vertical')


# # Distribution of monthly sales for store 1 including all departments

# In[64]:


fig, axarr=pl.subplots(4,3, sharex =True, sharey =True,figsize=(15,10))
s, m =1,1
for i in range(0,4):
    for j in range(0,3):
        xxx=axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['Store']== s) & (df_sales['Date'].dt.month ==m)],20,
                                                         density=True)
        axarr[i,j].set_yscale('log')
        axarr[i,j].set_xscale('log')
        axarr[i,j].set_ylim(1e-10,1e-4)
        axarr[i,j].set_xlim(5e2,1e6)
        axarr[i,j].set_title('%s'%calendar.month_name[m])
        m+=1
fig.text(0.5, 0.04, 'Weekly Sales', ha='center')
fig.text(0.04, 0.5, 'Number', va='center', rotation='vertical')        


# # Distribution of monthly sales for store 1 by 2010, 2011, 2012

# In[74]:


fig, axarr= pl.subplots(4,3, sharex =True,sharey=True,figsize=(15,10))
s,m =1,1
for i in range(0,4):
    for j in range(0,3):
        xxx=axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['Store']==s & (df_sales['Date'].dt.year ==2010)&
                           (df_sales['Date'].dt.month == m))], 20,color='b',histtype='step',density='True')
        xxx=axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['Store']==s & (df_sales['Date'].dt.year==2011)&
                                                         (df_sales['Date'].dt.month == m))], 20,color='g',
                           histtype='step',density='True')
        xxx=axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['Store']==s & (df_sales['Date'].dt.year==2012)& (
        df_sales['Date'].dt.month ==m))],20,color='r',histtype='step',density='True')
        axarr[i,j].set_yscale('log')
        axarr[i,j].set_xscale('log')
        axarr[i,j].set_ylim(1e-10,1e-4)
        axarr[i,j].set_xlim(5e2,1e6)
        axarr[i,j].set_title('%s'%calendar.month_name[m])
        m += 1
        
fig.text(0.5, 0.04, 'Weekly Sales', ha='center')
fig.text(0.04, 0.5, 'Number', va='center', rotation='vertical')        


# In[76]:


df_stores['SizeBand'] = pd.cut(df_stores['Size'], bins=4, labels=np.arange(1, 5)).astype(np.int)
#pandas.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise')
#Use cut when you need to segment and sort data values into bins. This function is also useful for going from a 
#continuous variable to a categorical variable
# For example, cut could convert ages to groups of age ranges.


# In[77]:


StoreSizeDict = df_stores.set_index('Store').to_dict()['SizeBand']
StoreTypeDict = df_stores.set_index('Store').to_dict()['Type'] #Convert the DataFrame to a dictionary


# In[78]:


df_sales['SizeBand'] = df_sales['Store']
df_sales['SizeBand'] = df_sales['SizeBand'].map(StoreSizeDict)
df_sales['Type'] = df_sales['Store'].map(StoreTypeDict)


# # Sum of weekly sales for month for 2010-2012

# In[83]:


fig, axarr=pl.subplots(4,3,sharex=True,sharey=True,figsize=(15,10))
s,m=1,1
for i in range(0,4):
    for j in range(0,3):
        dtf=df_sales[(df_sales['Store']==s)
                    & (df_sales['Date'].dt.month==m)
                    & (df_sales['Date'].dt.year==2010)]
        dtf=dtf.groupby('Date')['Weekly_Sales'].sum().reset_index()
        xxx=axarr[i,j].hist(dtf['Weekly_Sales'], 20, color='b');
        
        dtf=df_sales[(df_sales['Store']==s)
                    & (df_sales['Date'].dt.month==m)
                    & (df_sales['Date'].dt.year==2011)]
        dtf=dtf.groupby('Date')['Weekly_Sales'].sum().reset_index()
        xxx=axarr[i,j].hist(dtf['Weekly_Sales'], 20, color='g');
        
        dtf=df_sales[(df_sales['Store']==s)
                    & (df_sales['Date'].dt.month==m)
                    & (df_sales['Date'].dt.year==2012)]
        dtf=dtf.groupby('Date')['Weekly_Sales'].sum().reset_index()
        xxx=axarr[i,j].hist(dtf['Weekly_Sales'], 20, color ='r')
        
        axarr[i,j].set_title('%s'%calendar.month_name[m])
        axarr[i,j].set_xscale('log')
        axarr[i,j].set_xlim(5e5,5e6)
        m += 1
fig.text(0.5, 0.04, 'Sum of weekly Sales', ha='center')
fig.text(0.04, 0.5, 'Number', va='center', rotation='vertical')


# # Weekly sales by size of stores

# In[84]:


fig, axarr = pl.subplots(2, 2, sharex=True, sharey=True,figsize=(15,10))
s, m = 1, 1
for i in range(0, 2):
    for j in range(0, 2):
        xxx = axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['SizeBand'] == m)], 
                              20, density=True)
        axarr[i,j].set_yscale('log')
        axarr[i,j].set_xscale('log')
        axarr[i,j].set_xlim(5e2,1e6)
        axarr[i,j].set_title('Size Band = %d'%m)
        m += 1


# # Weekly sales by holiday weekends and size of the stores 

# # The distributions of weekly sales depends on whether or not the week will have a holiday or not

# In[86]:


fig, axarr = pl.subplots(2, 2, sharex=True, sharey=True,figsize=(15,10))
s, m = 1, 1
for i in range(0, 2):
    for j in range(0, 2):
        xxx = axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['SizeBand'] == m) & 
                             (df_sales['IsHoliday'] == False)], 50, color='b', 
                              density=True, histtype='step')
        xxx = axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['SizeBand'] == m) & 
                             (df_sales['IsHoliday'] == True)], 50, color='r', 
                              density=True, histtype='step')
        axarr[i,j].set_yscale('log')
        axarr[i,j].set_xscale('log')
        #axarr[i,j].set_ylim(1,1e4)
        axarr[i,j].set_xlim(5e2,1e6)

        m += 1
fig.text(0.5, 0.04, 'Weekly Sales', ha='center')
fig.text(0.04, 0.5, 'Normalized umber (Red/Blue:Holiday/Non)', va='center', rotation='vertical')


# # Distributions of weekly sales by size and holiday: 
# There is not difference between the distributions of weekly sale for small size stores even if we
# divided the stores by type. However, there is marginal differnce between these distributions for large sized stores

# In[87]:


fig, axarr = pl.subplots(4, 3, sharex=True, sharey=True,figsize=(15,10))
s, m = ['A', 'B', 'C'], 1
for i in range(0, 4):
    for j in range(0, 3):
        xxx = axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['SizeBand'] == m) 
                              & (df_sales['Type'] == s[j]) & (df_sales['IsHoliday'] == False)], 50, 
                              density=True, color='b', histtype='step');
        xxx = axarr[i,j].hist(df_sales['Weekly_Sales'].loc[(df_sales['SizeBand'] == m) 
                              & (df_sales['Type'] == s[j]) & (df_sales['IsHoliday'] == True)], 50, 
                              density=True, color='r', histtype='step');
        axarr[i,j].set_yscale('log')
        axarr[i,j].set_xscale('log')
        axarr[i,j].set_ylim(1e-10,1e-4)
        axarr[i,j].set_xlim(5e2,1e6)
        axarr[i,j].set_title('SizeBand=%d, Type=%s'%(m, s[j]))
        m += 1
fig.text(0.5, 0.04, 'Weekly Sales', ha='center')
fig.text(0.04, 0.5, 'Normalized umber (Red/Blue:Holiday/Non)', va='center', rotation='vertical')


# In[88]:


df_features = df_sales.merge(df_features, left_on=('Store', 'Date'), 
                                 right_on=('Store', 'Date'), how='left')


# In[89]:


df_features.columns.tolist()


# In[90]:


df_features.Type.unique()


# In[92]:


df_features = df_features.drop(['IsHoliday_y'], axis=1)
df_features = df_features.rename(columns = {'IsHoliday_x':'IsHoliday'})
df_features['IsHoliday'] = df_features['IsHoliday'].astype(int)


# In[93]:


# Compute the correlation matrix
corr = df_features.drop(['Store', 'Dept'], axis=1).corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


# # Dependence of MarkDown2 and MarDown3 on Holiday :
# 
# It is seen that there is a positive correlation between holiday weekends and markdown 2 and 3.
# There is positive correlation between the size of the store and the MarkDown 1, 4 and 4
# There is a positive correlation between size of the store and the weekly sales. However, it may be a systematic correlation as there are many employees needs to be worked for a larger store.
# There is a marginal positive correlation between the fuel price and MarkDown 1 but all other MarkDowns are negatively correlated with fuel price.
# Temperature of the region is mostly anticorrelated with the MarkDowns and no correlation between weekly sales
# CPI and unemployment are marginally anticorrelated with MarkDowns and no correlation between weekly sales

# In[94]:


#correlation matrix
f, ax = pl.subplots(figsize=(12, 9))
sns.heatmap(df_features.drop(['Store', 'Dept'], axis=1).corr(), mask=mask, vmax=.8, square=True);


# # How the type of stores have correlation:
# Similar conclusions for A and B type of stores can be derived. 
# There are slight difference between these conclusion and C type stores

# In[99]:


#correlation matrix
fig, axarr = pl.subplots(1,3, figsize=(15, 3.5))
sns.heatmap(df_features.drop(['Store', 'Dept'], axis=1)[df_features.Type == 'A'].corr(), mask=mask, vmax=.6, 
            square=True, ax=axarr[0], cbar=None);
sns.heatmap(df_features.drop(['Store', 'Dept'], axis=1)[df_features.Type == 'B'].corr(), mask=mask, vmax=.6, 
            square=True, ax=axarr[1], cbar=None);
sns.heatmap(df_features.drop(['Store', 'Dept'], axis=1)[df_features.Type == 'C'].corr(), mask=mask, vmax=.6, 
            square=True, ax=axarr[2], cbar=None);


# In[8]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
def mape(y_true, y_pred):
    ABS = np.abs(y_true - y_pred)
    return (ABS.sum()/y_true.sum()) * 100
seed = 123
model = RandomForestRegressor(n_estimators=20, criterion='mse', bootstrap=True, n_jobs=-1, random_state=seed)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
100-mape(np.expm1(X_test), np.expm1(y_pred))


# In[ ]:




