from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model

# read csv
df = pd.read_csv('uppdaterad.csv')

# define variable price per year, per square in Kronor mean
pris_per_år = df.groupby('år').Fastighetsprisbostadsrättkronorkvm.mean().reset_index()
#define X variable as year and reshape
X = pris_per_år['år']

# -1 keeps element as they are, 1 , a new dimension is added when you put in 1
X = X.values.reshape(-1, 1)
#define y as price  square foot in kronor
y = pris_per_år['Fastighetsprisbostadsrättkronorkvm']
# plot
plt.scatter(X,y)

print(X.ndim)
print(y.ndim)
print(y.shape)
print(X.shape)

# apply regression

regr = linear_model.LinearRegression()
#model fit
regr.fit(X,y)

# prediction
y_predict = regr.predict(X)
# plot prediction
plt.plot(X,y_predict)
X_future = np.array(range(2019,2025))
X_future= X_future.reshape(-1,1)
future_predict = regr.predict(X_future)
print(future_predict)
plt.plot(X_future,future_predict)
plt.show()