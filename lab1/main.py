import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#STEP 1
pds.set_option('display.max_columns', None)  # Показывать все столбцы
pds.set_option('display.width', 1000)  # Увеличить ширину вывода

customers = pds.read_csv("Ecommerce Customers.csv")

print(customers.head(7))
print("===============================")
print(customers.info())
print("===============================")
print(customers.describe())
print("===============================")

#STEP 3
#verify correlation for Time on Website and Yearly Amount Spent
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers, kind='scatter')
plt.show()

#STEP 4
#verify correlation for Time on App and Yearly Amount Spent
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers, kind='scatter')
plt.show()

#STEP 5
#copmaration of Time on App and Length of Membership (trand)
sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex')
plt.show()

#STEP 6
#interconnections in data set
sns.pairplot(customers[['Avg. Session Length',
                        'Time on App',
                        'Time on Website',
                        'Length of Membership',
                        'Yearly Amount Spent']], diag_kws={'bins': 10})
plt.show()

#STEP 7
#linear model
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
plt.show()

#STEP 8
#X and Y definitions
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

#STEP 9
#creation of a dataset and a testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#STEP 10, 11, 12
#model training

#lm = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
#copy_X=True, fit_intercept=True, n_jobs=1 / n_jobs=none, normalize=False / StandardScaler
lm = LinearRegression()
lm.fit(X_train, y_train)

print("Model coefficients:")
print(f"Avg. session length: {lm.coef_[0]}")
print(f"Time on App: {lm.coef_[1]}")
print(f"Time on Website: {lm.coef_[2]}")
print(f"Length of Membership: {lm.coef_[3]}")
print("===============================")

#STEP 14
#values predication for X_test
predictions = lm.predict(X_test)

#STEP 15
#scatter plot
plt.scatter(y_test, predictions)
plt.xlabel("Y Test")
plt.ylabel("Predicated Y")
plt.title("Scatter Plot")
plt.show()

#STEP 16
#errors
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)


# results
print(f"R^2: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

#STEP 17
residuals = y_test - predictions

# Строим гистограмму остатков
sns.histplot(residuals, bins=50, kde=True)
plt.xlim(-50, 50)
plt.xlabel("Yearly Amount Spent")
plt.show()