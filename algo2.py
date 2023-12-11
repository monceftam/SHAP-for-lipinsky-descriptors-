#!/usr/bin/env python
# coding: utf-8

# In[214]:


# Import required libraries
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from chembl_webresource_client.new_client import new_client
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
import shap


# In[215]:


# Initialize ChEMBL client
activity = new_client.activity
target = new_client.target


# In[253]:


# Step 1: Search for the HER2 target.
her2_query = target.search('INSR')
her2_targets = pd.DataFrame.from_dict(her2_query)


# In[254]:


# Display the list of search results
print("List of HER2 target search results:")
print(her2_targets[['target_chembl_id', 'organism', 'target_type', 'pref_name']])



# In[268]:


# Extract the HER2 target ChEMBL ID (assuming it's the first result)
her2_target_id = her2_targets['target_chembl_id'][0]


# In[269]:


# Step 2: Retrieve activities for the HER2 target
her2_activities = activity.filter(target_chembl_id=her2_target_id).filter(standard_type="IC50")


# In[270]:


# Convert the activities to a DataFrame
df = pd.DataFrame.from_dict(her2_activities)


# In[271]:


# Step 3: Data cleaning and preparation
# Convert 'standard_value' to numeric, handle errors, and drop missing values
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')


# In[272]:


# Remove rows with non-positive 'standard_value' as logarithm of zero or negative numbers is not meaningful
df = df[df['standard_value'] > 0]


# In[273]:


# Calculate pIC50 values
df['pIC50'] = -np.log10(df['standard_value'] * (10**-9))


# In[274]:


# Remove rows with any remaining NaN values in 'pIC50'
df = df.dropna(subset=['pIC50'])

# Display the first few rows of the cleaned DataFrame
print(df.head())


# In[275]:


# Update global descriptors
descriptors = ['MW', 'LogP', 'PSA', 'HBD', 'HBA']  # HBD: Hydrogen Bond Donors, HBA: Hydrogen Bond Acceptors

# Define a function to calculate the necessary descriptors
def calculate_descriptors(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str):
        return [np.nan] * len(descriptors)  # Return NaNs if the input is not a valid SMILES string

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        psa = rdMolDescriptors.CalcTPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        return [mw, logp, psa, hbd, hba]
    else:
        return [np.nan] * len(descriptors)



# In[276]:


# Fit a Gaussian mixture model to the pIC50 values and determine cutoffs
gmm = GaussianMixture(n_components=2, random_state=0).fit(df['pIC50'].values.reshape(-1, 1))
means = gmm.means_.flatten()
sorted_indices = np.argsort(means)
active_cutoff = means[sorted_indices[1]]
inactive_cutoff = means[sorted_indices[0]]


# In[277]:


# Visualization of pIC50 values
sns.histplot(df['pIC50'], bins=30, kde=False)
plt.axvline(x=active_cutoff, color='green', linestyle='--', label='Active Cutoff')
plt.axvline(x=inactive_cutoff, color='red', linestyle='--', label='Inactive Cutoff')
plt.xlabel('pIC50')
plt.ylabel('Frequency')
plt.title('Distribution of pIC50 values with Cutoffs')
plt.legend()
plt.show()


# In[278]:


# Split the dataset into active and inactive compounds
active_df = df[df['pIC50'] >= active_cutoff]
inactive_df = df[df['pIC50'] < inactive_cutoff]


# In[279]:


# Process and evaluate function
def process_and_evaluate(dataframe, compound_type):
    dataframe[descriptors] = dataframe['canonical_smiles'].apply(lambda x: pd.Series(calculate_descriptors(x)))
    dataframe.dropna(subset=descriptors, inplace=True)
    scaler = MinMaxScaler()
    dataframe[descriptors] = scaler.fit_transform(dataframe[descriptors])
    X = dataframe[descriptors]
    y = dataframe['pIC50']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{compound_type} Compounds - Mean Squared Error: {mse}")
    print(f"{compound_type} Compounds - R-squared: {r2}")
    print(f"{compound_type} Compounds - Cross-Validation Scores: {scores}")
    print(f"{compound_type} Compounds - Average Cross-Validation Score: {np.mean(scores)}")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    return model





# In[267]:


# Process and evaluate the active compounds
active_model = process_and_evaluate(active_df, "Active")

# Process and evaluate the inactive compounds
inactive_model = process_and_evaluate(inactive_df, "Inactive")



# In[ ]:





# In[ ]:





# In[ ]:




