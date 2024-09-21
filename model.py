import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load your dataset
data = pd.read_csv('Carbon Emission.csv')

# Define your features and target
X = data.drop(columns=['CarbonEmission'])
y = data['CarbonEmission']

# List of categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing: OneHotEncode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough'  # For other numeric columns, pass them as is
)

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model
with open('carbon_emission_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
