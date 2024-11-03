#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Installing libraries
import pandas as pd


# In[2]:


trxn = pd.read_csv(r'C:\Users\HP\Downloads\Retail_Data_Transactions.csv')


# In[3]:


trxn


# In[4]:


response = pd.read_csv(r'C:\Users\HP\Downloads\Retail_Data_Response.csv')


# In[5]:


response


# In[6]:


df= trxn.merge(response, on='customer_id', how='left')


# In[7]:


#features
df.dtypes


# In[8]:


df.shape


# In[9]:


df.head
df.tail()


# In[10]:


df.describe()


# In[11]:


#MISSING VALUES
df.isnull().sum()


# In[12]:


(31/125000)*100


# In[13]:


df=df.dropna()


# In[14]:


df


# In[15]:


# change dtypes
df['trans_date'] = pd.to_datetime(df['trans_date'])


# In[16]:


df['response']= df['response'].astype('int64')


# In[17]:


df


# In[18]:


set(df['response'])


# In[19]:


df.dtypes


# In[20]:


from scipy import stats
import numpy as np

# Assuming df is your DataFrame and 'tran_amount' is the column of interest
# Calculate z-scores
z_scores = np.abs(stats.zscore(df['tran_amount']))

# Set a threshold
threshold = 3

# Identify outliers
outliers = z_scores > threshold

# Print the outliers
print(df['tran_amount'][outliers])


# In[21]:


from scipy import stats
import numpy as np

# Assuming df is your DataFrame and 'tran_amount' is the column of interest
# Calculate z-scores
z_scores = np.abs(stats.zscore(df['response']))

# Set a threshold
threshold = 3

# Identify outliers
outliers = z_scores > threshold

# Print the outliers
print(df['response'][outliers])


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is already defined and contains the 'tran_amount' column
sns.boxplot(x= df['tran_amount'])
plt.show()


# In[23]:


# creating new columns
df['month']=df['trans_date'].dt.month


# In[24]:


df


# In[25]:


# which 3 months have had the highest transaction amounts?

monthly_Sales= df.groupby('month')['tran_amount'].sum()
monthly_sales= monthly_Sales.sort_value(ascending=False).reset_index().head(3)
monthly_sales


# In[29]:


# Which 3 months have had the highest transaction amounts?

# Group by month and sum the transaction amounts
monthly_sales = df.groupby('month')['tran_amount'].sum()

# Sort the monthly sales in descending order and get the top 3 months
top_3_months = monthly_sales.sort_values(ascending=False).reset_index().head(3)

# Print the top 3 months
print(top_3_months)


# In[30]:


# customer having highest number of orders

# Calculate the count of orders per customer
customer_counts = df['customer_id'].value_counts().reset_index()

# Rename the columns for clarity
customer_counts.columns = ['customer_id', 'num_orders']

# Print the result
print("Customer counts:")
print(customer_counts)


# In[31]:


customer_counts= df['customer_id']. value_counts().reset_index()
customer_counts.columns=['customer_id','count']
customer_counts


# In[32]:


# sort 
top_5_cus= customer_counts.sort_values(by='count', ascending=False).head(5)
top_5_cus


# In[ ]:


sns.barplot(x='customer_id',y='count',data=top_5_cus)


# In[33]:


customer_sales= df.groupby('customer_id')['tran_amount'].sum().reset_index()
customer_sales



# In[34]:


# Sort customer sales by transaction amount and get the top 5
top_5_sal = customer_sales.sort_values(by='tran_amount', ascending=False).head(5)

# Print the top 5 sales
print(top_5_sal)


# In[35]:


sns.barplot(x='customer_id',y='tran_amount',data=top_5_sal)


# # ADVANCED ANALYTICS¶
# 

# # Time Series Analysis

# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# In[37]:


import matplotlib.dates as mdates

df['month_year']= df['trans_date'].dt.to_period('M')


# In[38]:


monthly_sales = df.groupby('month_year')['tran_amount'].sum()


# In[39]:


df['month_year']= df['trans_date'].dt.to_period('M')
monthly_sales = df.groupby('month_year')['tran_amount'].sum()

monthly_sales.index= monthy_sales.index.to_timestamp()
plt.figure(figsize=(12,6))
plt.plot(monthly_sales.index,monthly_sales.values)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xlabel('Month-Year')
plt.xlabel('Sales')
plt.title('Monthly_Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show


# In[40]:


# Ensure the 'trans_date' column is in datetime format
df['trans_date'] = pd.to_datetime(df['trans_date'])

# Perform the operations
df['month_year'] = df['trans_date'].dt.to_period('M')
monthly_sales = df.groupby('month_year')['tran_amount'].sum()

# Convert PeriodIndex to Timestamp
monthly_sales.index = monthly_sales.index.to_timestamp()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values)

# Set the date format on the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))

plt.xlabel("Month-Year")
plt.ylabel("Sales")
plt.title("Monthly Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[42]:


df


# # Cohort Segmentation

# In[43]:


# Recency
recency = df.groupby('customer_id')['trans_date'].max()

# Frequency
frequency = df.groupby('customer_id')['trans_date'].count()

# Monetary
monetary = df.groupby('customer_id')['tran_amount'].sum()

# Combine
rfm = pd.DataFrame({'recency': recency, 'frequency': frequency, 'monetary': monetary})


# In[44]:


rfm


# # Churn Analysis

# In[45]:


# customer segmentation
def segment_customer(row):
    if row["recency"].year>=2012 and row['frequency']>=15 and row['monetary']>1000:
        return 'P0'
    elif (2011<=row['recency'].year<2012) and (10<row['frequency']<15) and (500<=row['monetary']<=1000):
        return 'P1'
    else:
        return 'P2'
rfm['Segment']= rfm.apply(segment_customer, axis=1)


# In[46]:


rfm


# # Churn Analysis

# In[47]:


# Count the number of churned and active customers
churn_counts= df['response']. value_counts()

#plot 
churn_counts.plot(kind='bar')


# # Analyzing top customers

# In[48]:


top_5_cus = monetary.sort_values(ascending=False).head(5).index

top_customers_df = df_merged[df_merged['customer_id'].isin(top_5_cus)]


top_customer_sales = top_customers_df.groupby(['customer_id', 'month_year'])['tran_amount'].sum().unstack(level=0)


top_customer_sales.plot(kind='line')


# In[49]:


import pandas as pd

# Assuming today's date for recency calculation
today = pd.to_datetime('2024-06-18')

# Convert trans_date to datetime
df['trans_date'] = pd.to_datetime(df['trans_date'])

# Recency
recency = (today - df.groupby('customer_id')['trans_date'].max()).dt.days

# Frequency
frequency = df.groupby('customer_id')['trans_date'].count()

# Monetary
monetary = df.groupby('customer_id')['tran_amount'].sum()

# Combine
rfm = pd.DataFrame({'recency': recency, 'frequency': frequency, 'monetary': monetary})

# Find the top 5 customers by monetary value
top_5_cus = monetary.sort_values(ascending=False).head(5).index

# Filter the original DataFrame to include only the top 5 customers
top_customers_df = df[df['customer_id'].isin(top_5_cus)]

# Group by customer_id and month_year and sum the transaction amounts
top_customer_sales = top_customers_df.groupby(['customer_id', 'month_year'])['tran_amount'].sum().unstack(level=0)

# Plot the sales of the top 5 customers over time
top_customer_sales.plot(kind='line')


# In[50]:


df


# In[53]:


df.to_csv('Maindata.csv')


# In[54]:


rfm.to_csv('addAnlys.csv')


# In[ ]:




