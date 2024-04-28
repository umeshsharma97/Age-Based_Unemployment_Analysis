#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\Umesh Sharma\Downloads\unemployment_rate_by_age_groups.csv")


# In[3]:


df


# In[4]:


df.describe()


# # Data Preprocessing:
# 

# In[5]:


df.isnull().sum()


# In[6]:


sns.heatmap(df.isnull())


# In[7]:


df.drop(["Area Type"],inplace=True,axis=1)


# In[8]:


df.head()


# In[9]:


numeric_cols = df.iloc[:, 4:]


# In[10]:


Q1=numeric_cols.quantile(0.25)
Q3=numeric_cols.quantile(0.75)
IQR=Q3-Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (numeric_cols < lower_bound) | (numeric_cols > upper_bound)
print(outliers)


# In[11]:


if outliers.any().any():
    print("Outliers exist in the DataFrame.")
else:
    print("No outliers found in the DataFrame.")


# # Exploratory Data Analysis (EDA):
# 

# In[12]:


avg_unemployment = df.iloc[:, 4:].mean()
avg_unemployment.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Age Group')
plt.ylabel('Average Unemployment Rate')
plt.title('Average Unemployment Rate by Age Group')
plt.xticks(rotation=45)
plt.show()


# In[13]:


# Monthly Unemployment Rate Distribution
df.iloc[:, 5:].boxplot(figsize=(10, 6))
plt.xlabel('Age Group')
plt.ylabel('Unemployment Rate')
plt.title('Monthly Unemployment Rate Distribution by Age Group')
plt.xticks(rotation=45)
plt.show()


# In[14]:


# Age Group Unemployment Rate Change
unemployment_change = df.iloc[:, 4:].diff()
unemployment_change.plot(figsize=(10, 8))
plt.xlabel('Date')
plt.ylabel('Change in Unemployment Rate')
plt.title('Month-to-Month Change in Unemployment Rate by Age Group')
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.axhline(y=0, color='black', linestyle=':')
plt.show()


# In[15]:


# Heatmap of Unemployment Rates by Age Group and Month
df_pivot = df.pivot_table(index='Month', columns='Year', values=df.columns[4:])
plt.figure(figsize=(12, 8))
sns.heatmap(df_pivot, cmap='YlGnBu')
plt.title('Unemployment Rates by Age Group and Month')
plt.xlabel('Year')
plt.ylabel('Month')
plt.show()


# In[16]:


# Bar Plot of Total Unemployment Rate by Year
total_unemployment_yearly = df.groupby('Year')[df.columns[4:]].sum().sum(axis=1)
total_unemployment_yearly.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Year')
plt.ylabel('Total Unemployment Rate')
plt.title('Total Unemployment Rate by Year')
plt.xticks(rotation=45)
plt.show()


# In[17]:


plt.figure(figsize=[6,6])
plt.title('Total Unemployment Rate by AGE')
avg_unemployment.plot(kind='pie', autopct='%1.f%%')


# In[18]:


avg_unemployment1 = df.iloc[:, 5:].mean()
plt.figure(figsize=[6,6])
plt.title('Total Unemployment Rate by AGE (20+)')
avg_unemployment1.plot(kind='pie', autopct='%1.f%%')


# In[19]:


numerical_features = df.select_dtypes(include=['int', 'float'])

# Compute the correlation matrix
correlation_matrix = numerical_features.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[ ]:




