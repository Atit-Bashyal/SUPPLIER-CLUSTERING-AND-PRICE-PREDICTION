#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:29:11 2019

@author: Atit Bashyal, Peeter Mansoor

Implementation of visualization, Data Preprocessing, K-means, LinearRegression and RandomForestRegressor
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from pandas.io.stata import StataReader

from matplotlib import pyplot as plt
from IPython.display import display
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split



from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering


from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics






'''load the data_set'''

df_c = pd.read_csv('ctrucks_clean.csv').set_index('supplier').fillna(0.00)
df_c.columns
df_c['supplier']=df_c.index


df_c.drop('Unnamed: 0',axis=1,inplace=True)



sns.relplot(x="supplier", y="nbags", col="year", dashes=False, markers=True, kind="dist", data=df_c)


g = sns.catplot("supplier", col="year", col_wrap=1,data=df_c,kind="count", height=3, aspect=6)
g.set_xticklabels(rotation=90)
plt.savefig('suppyear')



plt.figure(figsize=(10,10))
g = sns.FacetGrid(df_c,col="year")
g = g.map(sns.countplot, "supplier")
plt.savefig('suppyear')

price_ = pd.DataFrame(df_c[['supplier','price','year']].groupby('supplier','year')['price'].count())

print(price_)





'''exploration'''




plt.figure()
sns.pairplot(df_c,diag_kind = 'kde',height=4) # plot pairplot using pairplot function from sns
plt.savefig('pairplot.jpeg')


# Compute the correlation matrix
corr = df_rf.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


''' K menas for suppliers'''

# create new Dataframe with average of columns to be used in Kmeans
df_ca = df_c[ ['nbags', 'net_weight',  'moisture','nut_count', 'outturn', 'defective','avg_wpb']]
df_ca.columns

df_avg = df_ca.groupby('supplier').mean()
df_avg.columns
df_avg

df_avg.to_excel('avg.xlsx')






'Kmeans'

scaler = MinMaxScaler()
ddf_scaled = scaler.fit_transform(df_avg)
print(ddf_scaled.shape)

kmeans = KMeans(n_clusters=2, max_iter=600, init ='k-means++', algorithm = 'auto')
kmeans.fit(ddf_scaled)
kmeans.predict(ddf_scaled)
kmeans.cluster_centers_
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('green')
        elif l==1:
            cols.append('yellow')

    return cols
# Create the colors list using the function above
cols=pltcolor(kmeans.predict(ddf_scaled))

#phase plane with k-means result
plt.figure()
ax=plt.subplot()
ax.scatter(ddf_scaled[:,3], ddf_scaled[:,4], c = cols,s=30)
ax.scatter(kmeans.cluster_centers_[0][3], kmeans.cluster_centers_[0][4], s=200, c='r', marker='o')
ax.scatter(kmeans.cluster_centers_[1][3], kmeans.cluster_centers_[1][4], s=200, c='r', marker='o')
ax.set_xlabel('Normalized Nut count')
ax.set_ylabel('Normalized Outturn')

'''find the optimum supplier'''

sweet_df = ddf_scaled[:,3:5]
print(sweet_df)
kmeans.cluster_centers_.shape
extend_codebook = kmeans.cluster_centers_[:,3:5]
print(extend_codebook)
midpoint = np.array([(extend_codebook[0][0]+extend_codebook[1][0])/2,(extend_codebook[0][1]+extend_codebook[1][1])/2])
midpoint_extended = np.array([(extend_codebook[0][0]+extend_codebook[1][0])/2,(extend_codebook[0][0]+extend_codebook[1][0])/2])[:,np.newaxis][:,np.newaxis]

difference = sweet_df - midpoint_extended

#find the absolute distance from the difference, abs distance = ||difference||
abs_dist_extended = np.sqrt((difference)**2)
print(midpoint)

#reduce the 3-d absolute distance array back into a 2-d array
abs_dist = abs_dist_extended.sum(axis=2).T[:,0]

print(abs_dist.shape)
print(abs_dist)
np.mean(abs_dist)

def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('green')
        elif l==1:
            cols.append('red')

    return cols

def pltmark(lst):
    cols=[]
    for l in lst:
        if l<=np.median(lst):
            cols.append('D')
        elif l>=np.median(lst):
            cols.append('o')

    return cols
# Create the colors list using the function above
cols=pltcolor(kmeans.predict(ddf_scaled))
mar = pltmark(abs_dist)

plt.figure()
ax=plt.subplot()
for i in range(len(mar)):
    ax.scatter(ddf_scaled[:,3][i], ddf_scaled[:,4][i], c = cols[i],s=30,marker=mar[i])
#ax.scatter(kmeans.cluster_centers_[0][3], kmeans.cluster_centers_[0][4], s=200, c='r', marker='o')
#ax.scatter(kmeans.cluster_centers_[1][3], kmeans.cluster_centers_[1][4], s=200, c='r', marker='o')
ax.scatter(midpoint[0],midpoint[1],s=100,c='k',marker='o')
ax.set_xlabel('Normalized Nut count')
ax.set_ylabel('Normalized Outturn')



'''Random Forest'''

# add label from k-mean clustering classification as variable

df_rf =  pd.read_csv('ctrucks_clean.csv').fillna(0.00).set_index(['supplier'])
df_rf['label']= df_rf.index



# make a new column with label

def label(lst):
    cols=[]
    for l in lst:
        if l<=np.median(lst):
            cols.append(1)
        elif l>=np.median(lst):
            cols.append(0)

    return cols

label_ = label(abs_dist)

dic_label = {supplier : supplier_quality for supplier, supplier_quality  in zip(df_avg.index, label_)}

lab=[]
for i in df_rf['label']:
        lab.append(dic_label[i])

df_rf['label']= lab




df_rf = df_rf[['nbags', 'price', 'net_weight','moisture', 'nut_count', 'outturn', 'defective', 'avg_wpb','label']]


# make a test and train split

df_201617 = pd.DataFrame(df_rf[df_rf.price != 0.00])
print(df_201617)
df_2015 = pd.DataFrame(df_rf[df_rf.price == 0.00])
print(df_2015)

train_x = df_201617[['nbags', 'net_weight','moisture', 'nut_count', 'outturn',  'avg_wpb','label']]
train_y = df_201617[['price']]

test_x =  df_2015[['nbags', 'net_weight','moisture', 'nut_count', 'outturn', 'avg_wpb','label']]

df_201617.to_excel('rf.xlsx')



#Make a rfmodel object with  500 trees
rf = RF(n_estimators =500,  oob_score=True , random_state =0)

# fit and train the model using the training data
rf.fit(train_x , np.ravel(train_y))

rfpred=rf.predict(test_x)
rf_pred=rf.predict(train_x)

rf.score(train_x , np.ravel(train_y))

metrics.mean_squared_error(np.ravel(train_y), rf.predict(train_x))

df_2015['price']= rf.predict(test_x)
print(df_2015)


# Variable of Imporatnce
plt.figure()
ax1=plt.subplot(1,1,1) # create a subplot axis
bar_width = 0.28 # assign a width for each barplot

#plot the horizontal bar plots using barh function for all three target variables in the same subplot ax1
#first argument is points in the vertical axis where the bar are plotted
#second argument is value of bar hieght in the horizontal axis

#plot for model predecting size diversity
ax1.barh([x for x in range(1,8)],(rf.feature_importances_)*100,bar_width,align='center')
#create a list of environmental variable names to use as labels for ticks in vertical axis
ylab=[item.get_text() for item in ax1.get_yticklabels()] # list with empty strings. list length determined my the ticks seen in the plots
# replace empty strings at even list indexes by variable names in the same order as they are present in the dataframe column names
ylab[1]='nbags'
ylab[2]='net_weight'
ylab[3]='moisture'
ylab[4]='nut_count'
ylab[5]='outturn'
ylab[6]='avg_wpb'
ylab[7]='label'

#set the vertical axis tick labels using the list created above
ax1.set_yticklabels(ylab)
plt.savefig('variable_imp_rf')

'''linear regression'''


print(train_x.shape)
print(test_x.shape)
lin_reg = LinearRegression()
lin_reg.fit(train_x , np.ravel(train_y)) #training the algorithm

#To retrieve the intercept:
print(lin_reg.intercept_)
#For retrieving the slope:
print(lin_reg.coef_)

plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(train_y['price'])

coeff_df = pd.DataFrame(lin_reg.coef_, train_x.columns, columns=['Coefficient'])
coeff_df

train_pred  = lin_reg.predict(train_x)

df = pd.DataFrame({'Actual': np.ravel(train_y), 'Train_pred': lin_reg.predict(train_x)})
df.head(25)

# training error
print('Mean Absolute Error(Train):', metrics.mean_absolute_error(np.ravel(train_y), train_pred))
print('Mean Squared Error:', metrics.mean_squared_error(np.ravel(train_y), train_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.ravel(train_y), train_pred)))

y_pred = lin_reg.predict(test_x)


''' m5p model implemented in Jupyter notebook: m5p.ipynb'''


'''actual and predected price distributions linear regression and random forest'''


plt.figure()
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(y_pred)
plt.savefig('linear_dist_test')

plt.figure()
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(train_pred)
plt.savefig('linear_dist_train')


plt.figure()
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(train_y)
plt.savefig('dist_actual')


plt.figure()
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(rfpred)
plt.savefig('rf_dist_test')

plt.figure()
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(rf_pred)
plt.savefig('rf_dist_train')



'''MSE graph'''

dic_MSE = {'Random-Forest train MSE':metrics.mean_squared_error(np.ravel(train_y), rf.predict(train_x)), 'linear regression train MSE': metrics.mean_squared_error(np.ravel(train_y), train_pred) ,'M5ptrain MSE':5660.759201}
print(dic_MSE)
df_error = pd.DataFrame(dic_MSE,index=[1])
df_error.to_excel("error.xlsx")

plt.figure()
ax1=plt.subplot(1,1,1) # create a subplot axis
bar_width = 0.28 # assign a width for each barplot

#plot the horizontal bar plots using barh function for all three target variables in the same subplot ax1
#first argument is points in the vertical axis where the bar are plotted
#second argument is value of bar hieght in the horizontal axis

#plot for model predecting size diversity
ax1.barh([x for x in range(1,4)],df_error.loc[1],bar_width,align='center')
#create a list of environmental variable names to use as labels for ticks in vertical axis
ylab=[item.get_text() for item in ax1.get_yticklabels()] # list with empty strings. list length determined my the ticks seen in the plots
# replace empty strings at even list indexes by variable names in the same order as they are present in the dataframe column names
ylab[1]='Random-Forest'
ylab[3]='Linear-regresson'
ylab[5]='m5p'

#set the vertical axis tick labels using the list created above
ax1.set_yticklabels(ylab)
ax1.set_xlabel('MSE')
plt.savefig('error rates')
