import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('Placement_Prediction_data.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.fillna(0, inplace=True)

# Features & target
x = df.drop(['StudentId', 'PlacementStatus'], axis=1)
y = df['PlacementStatus']

# Label Encoding (ONLY FEATURES)
le_intern = LabelEncoder()
le_hack = LabelEncoder()

x.loc[:, 'Internship'] = le_intern.fit_transform(x['Internship'])
x.loc[:, 'Hackathon'] = le_hack.fit_transform(x['Hackathon'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=100
)

# Model
classify = RandomForestClassifier(n_estimators=100, criterion="entropy")
classify.fit(x_train, y_train)

# Prediction & accuracy
ypred = classify.predict(x_test)
accuracy = accuracy_score(y_test, ypred)
print("Accuracy:", accuracy)

# Save model
pickle.dump(classify, open('model.pkl', 'wb'))

# Load & test model
model = pickle.load(open('model.pkl', 'rb'))

sample = [[8,1,3,2,9,4.8,0,1,71,87,0]]
print("Placement Prediction:", model.predict(sample))