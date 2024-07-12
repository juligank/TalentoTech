import keras
fmnist = keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

import numpy as np
import matplotlib.pyplot as plt


index=1
np.set_printoptions(linewidth=320)
print(f'label:{training_labels[index]}')
print(f'Image:\n{training_images[index]}')
plt.imshow(training_images[index])

training_images = training_images / 255.0
test_images = test_images / 255.0

model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                                 keras.layers.Dense(128,activation='relu'),
                                 keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acurracy'])
model.summary()

history= model.fit(training_images, training_labels, epochs=10)

import pandas as pd
pd.DataFrame(history.history).plot(grid=True)

loss, accuracy = model.evaluate(training_images, training_labels)

print("perdida en el conjunto de entrenamiento ",loss)
print("precision en el conjunto de entrenamiento", accuracy)


loss, accuracy =- model.evaluate(test_images, test_labels)
print("perdida en el conjunto de Prueba ",loss)
print("precision en el conjunto de Prueba", accuracy)

index = 1
print(f'Label:{test_labels[index]}')
classification = model.predict(test_images[index:index+1])
print(f'Classification:\n {classification.reshape(-1,1)}')

