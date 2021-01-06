import numpy as np
import matplotlib.pyplot as plt
import tf as tf
from tensorflow.keras import layers, datasets, models
import cv2 as cv
from tensorflow.python.keras.utils.np_utils import to_categorical

(training_images, training_labels), (testing_images, testing_labels) = datasets.fashion_mnist.load_data()

training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

'''
for img in range(16):
    plt.subplot(4, 4, img+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[img], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[img]])

plt.show()
'''

model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

predictions = model.predict(testing_images)

model.save('image_classifier.model')

model = models.load_model('image_classifier.model')


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10),predictions_array, color="#777777")
    plt.ylim([0,1])
    prediction_label = np.argmax(predictions_array)

    thisplot[prediction_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 10
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], testing_labels, testing_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  testing_labels)
plt.show()
