from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()
X, y = housing.data, housing.target

type(housing)
housing.keys()
housing['data']
housing['target']
housing['target_names']
housing['DESCR']
housing['feature_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp_reg = MLPRegressor(hidden_layer_sizes=(5000),
                       activation='relu',
                       solver='adam',
                       max_iter=100,
                       random_state=42,
                       verbose=True)

mlp_reg.fit(X_train_scaled,y_train)

y_pred = mlp_reg.predict(X_train_scaled)

mse = mean_squared_error(y_test, y_pred)
print("Error cuadratico medio del modelo:", mse)

