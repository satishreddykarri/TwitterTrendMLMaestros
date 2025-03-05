import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "twitter_trending_data.csv"
df = pd.read_csv(file_path)

# Encode categorical features
le_country = LabelEncoder()
df['country_encoded'] = le_country.fit_transform(df['country'])

le_context = LabelEncoder()
df['context_encoded'] = le_context.fit_transform(df['context'])

le_name = LabelEncoder()
df['name_encoded'] = le_name.fit_transform(df['name'])

# Fill missing description values with 0
df['description'].fillna(0, inplace=True)

# Clustering for filtering relevant contexts per country
kmeans = KMeans(n_clusters=min(len(df['context'].unique()), 10), random_state=42, n_init=10)
df['context_cluster'] = kmeans.fit_predict(df[['context_encoded']])

def get_contexts_for_country(country):
    country_enc = le_country.transform([country])[0]
    return df[df['country_encoded'] == country_enc]['context'].unique().tolist()

# Train-test split
X = df[['country_encoded', 'context_encoded', 'description']]
y = df['name_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')

def predict_trending_topic(country, context):
    if country not in le_country.classes_ or context not in le_context.classes_:
        return "Error: Unrecognized country or context"
    
    country_enc = le_country.transform([country])[0]
    context_enc = le_context.transform([context])[0]
    description = df[df['country'] == country]['description'].max()
    
    pred_encoded = rf_model.predict([[country_enc, context_enc, description]])[0]
    return le_name.inverse_transform([pred_encoded])[0]

if __name__ == "__main__":
    print("Model trained successfully!")