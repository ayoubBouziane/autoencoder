from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np 
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

#create a model
input_img = Input(shape=(28,28,1))
#encoder
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
# need compile

# loading data set
(x_train, _),(x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train),28,28,1))
x_test = np.reshape(x_test, (len(x_test),28,28,1))

#add noisy
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#train autoencoder
sequence_autoencoder.fit(x_train_noisy, x_train,
	epochs=100,
	batch_size=128,
	shuffle=True,
	validation_data=(x_test_noisy,x_test))

decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20,4))
for i in xrange(n):
	#display original image
	ax = plt.subplot(2,n,i)
	plt.imshow(x_test_noisy[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	#display reconstruct image
	ax = plt.subplot(2, n, i+n)
	plt.imshow(decoded_imgs[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()