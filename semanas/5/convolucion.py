import numpy as np
import matplotlib.pyplot as plt

# Definir las señales de entrada
x = np.array([1, 2, 3, 4, 5])
h = np.array([-1, 5, 3, -2, -1])

# Realizar la convolución usando numpy
y = np.convolve(x, h, mode='full')

# Graficar las señales
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(x)
plt.title('Señal x')
plt.xlabel('Índice')
plt.ylabel('Amplitud')

plt.subplot(3, 1, 2)
plt.stem(h)
plt.title('Señal h')
plt.xlabel('Índice')
plt.ylabel('Amplitud')

plt.subplot(3, 1, 3)
plt.stem(y)
plt.title('Convolución de x y h')
plt.xlabel('Índice')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()
