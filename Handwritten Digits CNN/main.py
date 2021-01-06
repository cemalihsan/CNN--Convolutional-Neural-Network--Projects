from tensorflow.keras import datasets, layers, models
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print(train_images.shape)
# print(train_labels[0])

# Reshape the data to fit the model
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

train_images = train_images[:20000]
train_labels = train_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]


# One-Hot Encoding
train_label_one = to_categorical(train_labels)
test_label_one = to_categorical(test_labels)
'''
# Building CNN Model
model = models.Sequential()
model.add(layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
hist = model.fit(train_images, train_label_one, validation_data=(test_images, test_label_one), epochs=3)

model.save('image_classifier.model')
'''

model = models.load_model('image_classifier.model')

predictions = model.predict(test_images[:4])
# print(predictions)

print(np.argmax(predictions, axis=1))
print(test_labels[:4])

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    image = test_images[i]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')

plt.show()
