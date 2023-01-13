#!/usr/bin/env python
# coding: utf-8

# # <center>Movie Correlation Project

# In[115]:


# importing the packages we will use in this project
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None



# Now we need to read in the data
df = pd.read_csv(r'C:\Users\marlo\Desktop\Projects\data_analysis_Portfolio\movies.csv')


# In[7]:


# looking at the data
df.head()


# In[116]:


# looking to see if there is have any missing data
# Let's loop through the data and see if there is anything missing

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[117]:


df.isnull().sum()


# In[118]:


#mean of missing data
mean_score = df["score"].mean()
mean_votes = df["votes"].mean()
mean_budget = df["budget"].mean()
mean_gross = df["gross"].mean()
mean_runtime = df["runtime"].mean()

df = df.fillna({"score":mean_score,"votes":mean_votes, "budget":mean_budget, "gross":mean_gross, "runtime":mean_runtime})


# In[119]:


# Data Types for our columns

print(df.dtypes)


# In[ ]:





# In[120]:


# change data type of columns

df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
df


# In[169]:


# Order our Data a little bit to see

df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[168]:


# pd.set_option('display.max_rows', None)


# In[ ]:


# Drop any duplicates

df.drop_duplicates()


# In[175]:


# scater plot with budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earning')
plt.ylabel('Budget for flim')

plt.show


# In[173]:


df.head()


# In[177]:


# Plot budget vs gross using seaborn

sns.regplot(x="gross", y="budget", data=df, scatter_kws={'color': 'red'}, line_kws={'color':'blue'})


# In[ ]:


# Correlation Matrix between all numeric columns


# In[180]:


df.corr()


# In[181]:


df.corr(method ='pearson')


# In[182]:


df.corr(method ='kendall')


# In[183]:


df.corr(method ='spearman')


# In[184]:


# there is a high correlation between  budget and gross


# In[185]:


correlation_matrix = df.corr(method ='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[186]:


# Using factorize - this assigns a random numeric value for each unique categorical value

df.apply(lambda x: x.factorize()[0]).corr(method='pearson')


# In[187]:


correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[190]:


correlation_matrix.corr()


# In[188]:


correlation_mat = df.apply(lambda x: x.factorize()[0]).corr()

corr_pairs = correlation_mat.unstack()

print(corr_pairs)


# In[191]:


sorted_pairs = corr_pairs.sort_values(kind="quicksort")

print(sorted_pairs)


# In[192]:


# taking a look at the ones that have a high correlation (> 0.5)

strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

print(strong_pairs)


# In[193]:


# Looking at the top 15 compaies by gross revenue

CompanyGrossSum = df.groupby('company')[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[194]:


df.groupby(['company', 'year'])[["gross"]].sum()


# In[195]:


CompanyGrossSum = df.groupby(['company', 'year'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company','year'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[196]:


CompanyGrossSum = df.groupby(['company'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[197]:


plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[198]:


sns.swarmplot(x="rating", y="gross", data=df)


# In[203]:


sns.stripplot(x="rating", y="gross", data=df)


# In[ ]:




