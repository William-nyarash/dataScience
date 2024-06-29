import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

data = pd.read_csv('src /csv/Traffic%20accidents_2019_Leeds.csv')
filtered_data = data[data['1st Road Class'] != 'N/A']
# check for the number of columns
# print(data.columns)
# Identify columns with categorical data
categorical_columns = ['Casualty Severity', '1st Road Class','Lighting Conditions', 'Weather Conditions']

# Create LabelEncoder object
label_encoder = LabelEncoder()

# Encode categorical columns
for column in categorical_columns:
  filtered_data[column] = label_encoder.fit_transform(filtered_data[column])

# Select features and target variable
features = filtered_data[['Casualty Severity', '1st Road Class','Lighting Conditions', 'Weather Conditions']]
target = filtered_data['Vehicle Number']
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
print(model.score(X_test, y_test))
