#!/usr/bin/env python
# coding: utf-8

# # Visual Reports

# In[3]:


import pandas as pd

# Load the CSV file into a pandas DataFrame
main_data_path = r'C:\Users\HP\Downloads\Maindata (1).csv'
main_data_df = pd.read_csv(main_data_path)

# Display the first few rows of the DataFrame to inspect the data
print(main_data_df.head())


# In[4]:


sales_by_month = main_data_df.groupby('month_year')['tran_amount'].sum().reset_index()
sales_by_month.columns = ['Month-Year', 'Total Transaction Amount']


# In[5]:


sales_by_customer = main_data_df.groupby('customer_id')['tran_amount'].sum().reset_index()
sales_by_customer.columns = ['Customer ID', 'Total Transaction Amount']


# In[6]:


response_rates_by_month = main_data_df.groupby('month_year')['response'].mean().reset_index()
response_rates_by_month.columns = ['Month-Year', 'Response Rate']


# In[7]:


#Python/Matplotlib/Seaborn Code
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_by_month, x='Month-Year', y='Total Transaction Amount')
plt.title('Sales Trend Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Total Transaction Amount')
plt.xticks(rotation=45)
plt.show()


# In[8]:


#Top Customers by Sales

top_customers = sales_by_customer.nlargest(10, 'Total Transaction Amount')

plt.figure(figsize=(12, 6))
sns.barplot(data=top_customers, x='Customer ID', y='Total Transaction Amount')
plt.title('Top 10 Customers by Sales')
plt.xlabel('Customer ID')
plt.ylabel('Total Transaction Amount')
plt.xticks(rotation=45)
plt.show()


# In[9]:


#Monthly Response Rates

plt.figure(figsize=(12, 6))
sns.barplot(data=response_rates_by_month, x='Month-Year', y='Response Rate')
plt.title('Monthly Response Rates')
plt.xlabel('Month-Year')
plt.ylabel('Response Rate')
plt.xticks(rotation=45)
plt.show()


# In[10]:


import pandas as pd

# Load the CSV file into a pandas DataFrame
data_path = r'C:\Users\HP\Downloads\addAnlys (1).csv'
data_df = pd.read_csv(data_path)

# Display the first few rows of the DataFrame to inspect the data
print(data_df.head())


# In[12]:


import pandas as pd

# Load the CSV file into a pandas DataFrame
data_path = r'C:\Users\HP\Downloads\addAnlys (1).csv'
data_df = pd.read_csv(data_path)

# List all columns in the DataFrame to understand its structure
print(data_df.columns)


# In[15]:


# List all columns in the DataFrame
print(data_df.columns)


# In[18]:


total_monetary = data_df[['customer_id', 'monetary']]


# In[19]:


frequency_by_customer = data_df[['customer_id', 'frequency']]


# In[20]:


recency_by_customer = data_df[['customer_id', 'recency']]


# In[21]:


#Distribution of Monetary Value
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.histplot(data_df['monetary'], bins=20, kde=True)
plt.title('Distribution of Monetary Values')
plt.xlabel('Monetary Value')
plt.ylabel('Frequency')
plt.show()


# In[22]:


#Distribution of Purchase Frequency

plt.figure(figsize=(12, 6))
sns.histplot(data_df['frequency'], bins=20, kde=True)
plt.title('Distribution of Purchase Frequency')
plt.xlabel('Frequency of Purchases')
plt.ylabel('Frequency')
plt.show()


# In[23]:


#Distribution of Recency
plt.figure(figsize=(12, 6))
sns.histplot(data_df['recency'], bins=20, kde=True)
plt.title('Distribution of Recency Values')
plt.xlabel('Recency (days since last purchase)')
plt.ylabel('Frequency')
plt.show()


# In[24]:


#Top Customers by Monetary Value
top_customers = data_df.nlargest(10, 'monetary')

plt.figure(figsize=(12, 6))
sns.barplot(data=top_customers, x='customer_id', y='monetary')
plt.title('Top 10 Customers by Monetary Value')
plt.xlabel('Customer ID')
plt.ylabel('Monetary Value')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




