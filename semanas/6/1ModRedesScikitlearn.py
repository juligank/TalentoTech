from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Carga del conjunto de datos Iris
iris = load_iris()
X, y = iris.data, iris.target

# Exploración básica del conjunto de datos Iris (opcional, solo para referencia)
type(iris)  # Tipo de objeto
iris.keys()  # Llaves del diccionario que contiene los datos
iris['data']  # Datos de características
iris['target']  # Etiquetas de las clases
iris['target_names']  # Nombres de las clases
iris['DESCR']  # Descripción del conjunto de datos
iris['feature_names']  # Nombres de las características

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarización de las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajusta y transforma el conjunto de entrenamiento
X_test_scaled = scaler.transform(X_test)  # Transforma el conjunto de prueba con el mismo ajuste

# Inicialización del clasificador de perceptrón multicapa (MLP)
mlp_clp = MLPClassifier(hidden_layer_sizes=(100),  # Una capa oculta con 100 neuronas
                        activation='relu',  # Función de activación ReLU
                        solver='adam',  # Optimizador Adam
                        max_iter=100,  # Número máximo de iteraciones
                        random_state=42,  # Semilla para reproducibilidad
                        verbose=True  # Muestra mensajes durante el ajuste
                        )

# Entrenamiento del modelo con los datos de entrenamiento escalados
mlp_clp.fit(X_train_scaled, y_train)

# Predicción de las etiquetas del conjunto de prueba
y_pred = mlp_clp.predict(X_test_scaled)

# Impresión de las etiquetas reales y predichas (para comparación)
print(y_test)
print(y_pred)

# Evaluación de la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)


