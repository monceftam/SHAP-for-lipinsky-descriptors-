#!/usr/bin/env python
# coding: utf-8

# In[217]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from keras.models import Sequential
from keras.layers import Dense, Dropout
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# In[210]:


# Load the dataset
df = pd.read_csv('HER2_compounds_pIC50_clean.csv')


# In[211]:


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


# In[202]:


# Process and evaluate function
def process_and_evaluate(dataframe, compound_type):
    # Calculate descriptors for each row in the dataframe and handle NaN values
    descriptor_data = dataframe['canonical_smiles'].apply(calculate_descriptors)
    descriptor_df = pd.DataFrame(descriptor_data.tolist(), columns=descriptors)

    # Combine original dataframe with descriptor data
    dataframe = pd.concat([dataframe.reset_index(drop=True), descriptor_df], axis=1)

    # Drop rows with NaN in descriptor columns
    dataframe.dropna(subset=descriptors, inplace=True)

    # Normalize descriptor data
    scaler = MinMaxScaler()
    dataframe[descriptors] = scaler.fit_transform(dataframe[descriptors])

    # Split data into training and test sets
    X = dataframe[descriptors]
    y = dataframe['pIC50']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model's performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{compound_type} Compounds - Mean Squared Error: {mse}")
    print(f"{compound_type} Compounds - R-squared: {r2}")

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{compound_type} Compounds - Cross-Validation Scores: {scores}")
    print(f"{compound_type} Compounds - Average Cross-Validation Score: {np.mean(scores)}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Prediction')
    plt.title(f'Actual vs Predicted pIC50 Values for {compound_type} Compounds')
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    return model



# In[203]:


# Apply the process_and_evaluate function to the whole dataset
model = process_and_evaluate(df, "All")


# In[204]:


# Updated process and evaluate function with hyperparameter tuning and feature selection
def process_and_evaluate(dataframe, compound_type, descriptors):
    
    # Calculate descriptors for each row in the dataframe and handle NaN values
    descriptor_data = dataframe['canonical_smiles'].apply(calculate_descriptors)
    descriptor_df = pd.DataFrame(descriptor_data.tolist(), columns=descriptors)

    # Combine original dataframe with descriptor data
    dataframe = pd.concat([dataframe.reset_index(drop=True), descriptor_df], axis=1)

    # Drop rows with NaN in descriptor columns
    dataframe.dropna(subset=descriptors, inplace=True)

    # Split data into features and target
    X = dataframe[descriptors]
    y = dataframe['pIC50']
    
    # Define a pipeline with scaling, feature selection, and the regressor
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Feature scaling
        ('feature_selection', SelectKBest(f_regression)),  # Feature selection
        ('regressor', RandomForestRegressor(random_state=42))  # Model
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'feature_selection__k': [2, 3],  # Number of features to select
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 4, 6]
    }
    
    # Define GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)  # Use all data for fitting
    
    # Output best hyperparameters
    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Split data into training and test sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluate the best model's performance
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Output performance metrics
    print(f"{compound_type} Compounds - Mean Squared Error: {mse}")
    print(f"{compound_type} Compounds - R-squared: {r2}")
    
    # Perform and output cross-validation scores
    cv_scores = cross_val_score(best_model, X, y, cv=5)
    print(f"{compound_type} Compounds - Cross-Validation Scores: {cv_scores}")
    print(f"{compound_type} Compounds - Average Cross-Validation Score: {np.mean(cv_scores)}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Prediction')
    plt.title(f'Actual vs Predicted pIC50 Values for {compound_type} Compounds')
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate and plot SHAP values
    explainer = shap.TreeExplainer(best_model.named_steps['regressor'])
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    return best_model


# In[205]:


# Assuming you have already defined your descriptors list like this:
descriptors = ['MW', 'LogP', 'PSA', 'HBD', 'HBA']  # Update this list based on your requirements

# Now, call the function with the correct number of arguments
best_model = process_and_evaluate(df, "All", descriptors)


# In[206]:


# Define the neural network structure
def create_neural_network(n_features):
    model = Sequential()
    model.add(Dense(64, input_dim=n_features, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# In[207]:


# Updated process and evaluate function with hyperparameter tuning and feature selection
def process_and_evaluate(dataframe, compound_type):
    # Calculate descriptors and drop missing values
    dataframe[descriptors] = dataframe['canonical_smiles'].apply(lambda x: pd.Series(calculate_descriptors(x)))
    dataframe.dropna(subset=descriptors, inplace=True)

    # Split data into features and target
    X = dataframe[descriptors]
    y = dataframe['pIC50']

    # Normalize the descriptor data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

    # Gradient Boosting
    gb_model = GradientBoostingRegressor()
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 4],
        'learning_rate': [0.01, 0.1]
    }
    grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_gb_model = grid_search.best_estimator_

    y_pred_gb = best_gb_model.predict(X_test)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    print(f"Gradient Boosting - Mean Squared Error: {mse_gb}")
    print(f"Gradient Boosting - R-squared: {r2_gb}")

    # Neural Network
    nn_model = create_neural_network(X_train.shape[1])
    history = nn_model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=1)

    y_pred_nn = nn_model.predict(X_test)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)
    print(f"Neural Network - Mean Squared Error: {mse_nn}")
    print(f"Neural Network - R-squared: {r2_nn}")

    # ... [Add any additional steps or model evaluations you need here] ...

    return best_gb_model, nn_model


# In[208]:


# Use the function to process and evaluate your data
best_gb_model, nn_model = process_and_evaluate(df, "All")


# In[222]:


# Function to print the details of the KMeans model
def print_kmeans_model_details(kmeans_model):
    # Print out the main parameters of the KMeans model
    print("KMeans Model Details:")
    print(kmeans_model)

    # Print the cluster centers (centroids)
    print("\nCluster Centers (Centroids):")
    print(kmeans_model.cluster_centers_)


# In[223]:


# Function to process and evaluate clustering
def process_and_evaluate_clustering(dataframe, compound_type, n_clusters):
    # Calculate descriptors for each row in the dataframe and handle NaN values
    descriptor_data = dataframe['canonical_smiles'].apply(calculate_descriptors)
    descriptor_df = pd.DataFrame(descriptor_data.tolist(), columns=descriptors)

    # Combine original dataframe with descriptor data
    dataframe = pd.concat([dataframe.reset_index(drop=True), descriptor_df], axis=1)

    # Drop rows with NaN in descriptor columns
    dataframe.dropna(subset=descriptors, inplace=True)

    # Normalize descriptor data
    scaler = MinMaxScaler()
    dataframe_normalized = scaler.fit_transform(dataframe[descriptors])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(dataframe_normalized)
    
    # Assign clusters to the original dataframe
    dataframe['cluster'] = kmeans.labels_

    # Evaluate clustering using silhouette score
    silhouette_avg = silhouette_score(dataframe_normalized, kmeans.labels_)
    print(f"{compound_type} Compounds - Silhouette Score: {silhouette_avg}")

    # Plot clustering results
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe_normalized[:, 0], dataframe_normalized[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title(f'Cluster Plot for {compound_type} Compounds')
    plt.xlabel('Normalized Descriptor 1')
    plt.ylabel('Normalized Descriptor 2')
    plt.colorbar(label='Cluster Label')
    plt.show()

    # Print the details of the model using the previously defined function
    print_kmeans_model_details(kmeans)

    return kmeans


# In[225]:


# Set compound type and number of clusters
compound_type = "All Compounds"
n_clusters = 3

# Perform clustering and print model details
kmeans_model = process_and_evaluate_clustering(df, compound_type, n_clusters)


# In[ ]:




