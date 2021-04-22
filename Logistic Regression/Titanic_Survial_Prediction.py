import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('titanic.csv')

# Update sex column to numerical
passengers["Sex"] = passengers["Sex"].apply(lambda row: 0 if row == "male" else 1)

# Fill the nan values in the age column



passengers['Age'].fillna(inplace=True,value=passengers['Age'].mean())

# Create a first class column

passengers["FirstClass"] = passengers["Pclass"].apply(lambda row: 1 if row == 1 else 0)

# Create a second class column
passengers["SecondClass"] = passengers["Pclass"].apply(lambda row: 1 if row == 2 else 0)
print(passengers)
# Select the desired features
features = passengers[['Sex','Age','FirstClass','SecondClass']]
survival =passengers[['Survived']]
# Perform train, test, split

X_train,X_test,y_train,y_test = train_test_split(features,survival,test_size=0.2)
# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
# Create and train the model
model = LogisticRegression()
model.fit(X_train,y_train)

# Score the model on the train data

print(model.score(X_train,y_train))
# Score the model on the test data

print(model.score(X_test,y_test))
# Analyze the coefficients, vilka värden vägde mest för att modellen skulle kunna göra en prediction.
print(model.coef_)


# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Astrid = np.array([1.0,31.0,0.0,1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack,Rose,Astrid])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)


# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
