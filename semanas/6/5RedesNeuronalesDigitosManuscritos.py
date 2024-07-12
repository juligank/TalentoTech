import keras
mnist = keras.datasets.mnist

(trainig_images, training_labels), (test_images, test_labels) = mnist.load_data()
import numpy as np
import matplotlib.pyplot as plt

index=1
np.set_printoptions(linewidth=320)
print(f'label:{training_labels[index]}')
print(f'Image:\n{trainig_images[index]}')
plt.imshow(trainig_images[index])

trainig_images = trainig_images / 255.0
test_images = test_images / 255.0

model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                                keras.layers.Dense(128, activation='relu'),
                                keras.layers.Dense(10, activation='softmax')
                                 ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(trainig_images, training_labels, epochs=10)

import pandas as pd
pd.DataFrame(history.history).plot(grid=True)

loss, accuracy = model.evaluate(trainig_images, training_labels)
print("perdida en el conjunto de entrenamiento", loss)
print("Precision en el conjunto de entrenamiento", accuracy)

loss, accuracy = model.evaluate(test_images, test_labels)
print("perdida en el conjunto de pruebas",loss)
print("precision en el conjunto de pruebas",accuracy)


