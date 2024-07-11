import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Obtener datos de vivienda de California
housing = fetch_california_housing()

# Dividir datos en entrenamiento y prueba
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

# Escalar datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

# Definir modelo de red neuronal
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['RootMeanSquaredError', 'MeanAbsolutePercentageError'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

# Crear DataFrame con el historial de entrenamiento
dfHistory = pd.DataFrame(history.history)

# Graficar pérdida y métricas de error
dfHistory[['loss','val_loss', 'root_mean_squared_error', 'val_root_mean_squared_error']].plot(grid=True)
plt.title('Pérdida y RMSE durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Valor')
plt.show()

dfHistory[['mean_absolute_percentage_error','val_mean_absolute_percentage_error']].plot(grid=True)
plt.title('MAPE durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Valor')
plt.show()

# Evaluar el modelo en el conjunto de entrenamiento
loss, RMSE, MAPE = model.evaluate(X_train, y_train)
print("Evaluar el modelo en el conjunto de entrenamiento:")
print("Pérdida (MSE) en entrenamiento:", loss)
print("RMSE en entrenamiento:", RMSE)
print("MAPE en entrenamiento:", MAPE)

# Evaluar el modelo en el conjunto de validación
loss, RMSE, MAPE = model.evaluate(X_valid, y_valid)
print("\nEvaluar el modelo en el conjunto de validación:")
print("Pérdida en validación:", loss)
print("RMSE en validación:", RMSE)
print("MAPE en validación:", MAPE)

# Evaluar el modelo en el conjunto de prueba
loss, RMSE, MAPE = model.evaluate(X_test, y_test)
print("\nEvaluar el modelo en el conjunto de prueba:")
print("Pérdida en prueba:", loss)
print("RMSE en prueba:", RMSE)
print("MAPE en prueba:", MAPE)
