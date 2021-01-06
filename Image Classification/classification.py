import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255  # normalizing images between 0 to 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# for image in range(16):
#     plt.subplot(4, 4, image + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[image], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[image][0]])
#
# plt.show()

# Restricting sample size to 20000
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_labels = testing_labels[:4000]
testing_images = testing_images[:4000]


# Open Comments If Training is Needed for Neural Network
'''
model = models.Sequential() #Creates layer by layer models(linear stack of layers) 
#convolution kernel with the layer input to produce a tensor of outputs 
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) 
#1st Arg: learn a total of filter, 2nd Arg: Kernel size(2D convolution window,MUST BE ODD NUMBER)
#3rd Arg: reLU(rectified linear activation function) if input > 0 return input, else return 0.

model.add(layers.MaxPooling2D((2, 2))) # elects the maximum element from the region of the feature map, reduce to 2x2
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation='relu')) # 64 neurons in first hidden layer
model.add(layers.Dense(10, activation='softmax')) 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model') # saves neural network model
'''

model = models.load_model('image_classifier.model')

img = cv.imread('deer.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction is {class_names[index]}")

plt.show()
