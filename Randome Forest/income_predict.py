
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

income_data = pd.read_csv('income.csv')

income_data["sex-int"] = income_data["Sex"].apply(lambda row: 0 if row == "Male" else 1)

income_data["native-country-int"] = income_data["Native_country"].apply(lambda row: 0 if row == "United-States" else 1)



labels = income_data[["Income"]]
data = income_data[['Age','Capital_gain','Capital_loss','Hours_per_week', 'sex-int',"native-country-int"]]



train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

forest = RandomForestClassifier(random_state = 1)

forest.fit(train_data,train_labels)

print(forest.score(test_data,test_labels))