# Importación de librerías necesarias
import numpy as np  # Librería para trabajar con arrays
import keras  # Librería para construir y entrenar redes neuronales

# Definición del modelo secuencial de Keras
model = keras.Sequential([
    # Añadir una capa densa con una neurona y una entrada de una dimensión
    keras.layers.Dense(units=1, input_shape=[1])
])

# Compilación del modelo
# - Optimizer: 'sgd' (Stochastic Gradient Descent)
# - Loss: 'mean_squared_error' (Error cuadrático medio)
model.compile(optimizer='sgd', loss='mean_squared_error')

# Definición de los datos de entrada (X) y los datos de salida (y)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
y = np.array([11.0, 21.0, 31.0, 41.0, 51.0], dtype=float)

# Entrenamiento del modelo
# - epochs: número de iteraciones sobre el conjunto de datos
model.fit(X, y, epochs=100)

# Predicción de la salida para un nuevo valor de entrada
print(model.predict([10.0]))


