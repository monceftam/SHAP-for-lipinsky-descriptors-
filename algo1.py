#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client


# In[2]:


# Initialize ChEMBL client
activity = new_client.activity
target = new_client.target


# In[3]:


# Step 1: Search for the HER2 target.
her2_query = target.search('INSR')
her2_targets = pd.DataFrame.from_dict(her2_query)


# In[4]:


# Display the list of search results
print("List of HER2 target search results:")
print(her2_targets[['target_chembl_id', 'organism', 'target_type', 'pref_name']])


# In[5]:


# Extract the HER2 target ChEMBL ID (assuming it's the first result)
her2_target_id = her2_targets['target_chembl_id'][0]


# In[6]:


# Step 2: Retrieve activities for the HER2 target
her2_activities = activity.filter(target_chembl_id=her2_target_id).filter(standard_type="IC50")


# In[7]:


# Convert the activities to a DataFrame
df = pd.DataFrame.from_dict(her2_activities)


# In[8]:


# Step 3: Data cleaning and preparation
# Convert 'standard_value' to numeric, handle errors, and drop missing values
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')


# In[9]:


# Remove rows with non-positive 'standard_value' as logarithm of zero or negative numbers is not meaningful
df = df[df['standard_value'] > 0]


# In[10]:


# Calculate pIC50 values
df['pIC50'] = -np.log10(df['standard_value'] * (10**-9))


# In[11]:


# Remove rows with any remaining NaN values in 'pIC50'
df = df.dropna(subset=['pIC50'])

# Save the cleaned DataFrame to a CSV file
df.to_csv('HER2_compounds_pIC50_clean.csv', index=False)

# Display the first few rows of the cleaned DataFrame
print(df.head())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




