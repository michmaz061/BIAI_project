import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=50)
# model.save('num_reader.model')

model = tf.keras.models.load_model('num_reader.model')
loss, acc = model.evaluate(x_test, y_test)
image_number = 1
while os.path.isfile(f"dig/digit_{image_number}.jpg"):
    try:
        image = cv.imread(f"dig/digit_{image_number}.jpg")[:, :, 0]
        image = np.invert(np.array([image]))
        prediction = model.predict(image)
        print("{digit_", image_number, ")", f"Prediction of number is: {np.argmax(prediction)}")
        plt.imshow(image[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number += 1
