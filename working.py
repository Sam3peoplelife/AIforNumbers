import keras.models
import keras.utils as image
import matplotlib.pyplot as plt
import numpy as np

# loading models
model1 = keras.models.load_model('Hopfield network')
model2 = keras.models.load_model('Convolutional neural network')

# path name
filename = "image.jpg"

# loading and transforming image
img = image.load_img(filename, grayscale=True)
img_array = image.img_to_array(img)
inverted_array = 255 - img_array
inverted_array = inverted_array.astype('float32') / 255
inverted_array = np.array(inverted_array.reshape(1, 28, 28))

# directory with predictions
predictions = {
    'Model1': np.argmax(model1.predict(inverted_array)),
    'Model2': np.argmax(model2.predict(inverted_array))
}

print(predictions)

encoder = keras.models.load_model('Encoder')
decoder = keras.models.load_model('Decoder')
def show_img():
    ax1 = plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.gray()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2 = plt.subplot(2, 1, 2)
    plt.imshow(img_rep.reshape(28, 28))
    plt.title('Autoencoder')
    plt.gray()
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    plt.show()

inverted_array = inverted_array.reshape((len(inverted_array), np.prod(inverted_array.shape[1:])))
img_rep = decoder.predict(encoder.predict(inverted_array))
show_img()



