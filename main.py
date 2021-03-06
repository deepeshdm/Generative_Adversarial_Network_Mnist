import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Reshape, \
    BatchNormalization, Dropout, Flatten, Dense, Activation, UpSampling2D


# ---------------------------------------------------------------

def discriminator():
    net = Sequential()
    input_shape = (28, 28, 1)
    dropout_prob = 0.5

    net.add(Conv2D(64, kernel_size=4, strides=2, input_shape=input_shape, padding='same'))
    net.add(LeakyReLU())
    net.add(BatchNormalization())

    net.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    net.add(LeakyReLU())
    net.add(BatchNormalization())
    net.add(Dropout(dropout_prob))

    net.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    net.add(LeakyReLU())
    net.add(BatchNormalization())
    net.add(Dropout(dropout_prob))

    net.add(Conv2D(512, kernel_size=4, strides=2, padding='same'))
    net.add(LeakyReLU())
    net.add(BatchNormalization())
    net.add(Dropout(dropout_prob))

    net.add(Flatten())
    net.add(Dense(1))
    net.add(Activation('sigmoid'))

    return net


def generator():
    net = Sequential()
    dropout_prob = 0.5

    net.add(Dense(7 * 7 * 256, input_dim=100))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(Reshape((7, 7, 256)))
    net.add(Dropout(dropout_prob))

    net.add(UpSampling2D(size=(2, 2)))
    net.add(Conv2D(128, kernel_size=(2, 2), strides=2))
    net.add(BatchNormalization())
    net.add(ReLU())

    net.add(UpSampling2D(size=(2, 2)))
    net.add(Conv2D(64, kernel_size=(2, 2), strides=2))
    net.add(BatchNormalization())
    net.add(ReLU())

    net.add(UpSampling2D(size=(2, 2)))
    net.add(Conv2D(32, 2, padding='same'))
    net.add(BatchNormalization())
    net.add(ReLU())

    net.add(UpSampling2D(size=(2, 2)))
    net.add(Conv2D(16, 2, padding='same'))
    net.add(BatchNormalization())
    net.add(ReLU())

    net.add(Conv2D(1, 2, padding='same'))
    net.add(Activation('tanh'))

    return net


# ---------------------------------------------------------------

# We create & add the discriminator network to a new Sequential
# model and do not directly compile the discriminator itself.

net_discriminator = discriminator()
optim_discriminator = Adam(learning_rate=0.0002, beta_1=0.5)
model_discriminator = Sequential()
model_discriminator.add(net_discriminator)
model_discriminator.compile(loss='binary_crossentropy', optimizer=optim_discriminator, metrics=['accuracy'])

model_discriminator.summary()

# ---------------------------------------------------------------

# We add both networks to a combined model i.e adversarial model.
# Also we freeze the discriminator in this model.

net_generator = generator()
optim_adversarial = Adam(learning_rate=0.0002, beta_1=0.5)
model_adversarial = Sequential()
model_adversarial.add(net_generator)

# Disable layers in discriminator
for layer in net_discriminator.layers:
    layer.trainable = False

model_adversarial.add(net_discriminator)
model_adversarial.compile(loss='binary_crossentropy', optimizer=optim_adversarial, metrics=['accuracy'])

model_adversarial.summary()

# ---------------------------------------------------------------

from keras.datasets import mnist

(x_train, _), (_, _) = mnist.load_data()

# scaling x_train between
x_train = x_train / 255

x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)

print(x_train.shape)

# ---------------------------------------------------------------

# Training the DCGAN model
batch_size = 128
iter = 10000

for i in range(iter):

    # Select a random set of training images from the mnist dataset
    images_train = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
    # Generate a random noise vector
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    # Use the generator to create fake images from the noise vector
    images_fake = net_generator.predict(noise)

    # Create a dataset with fake and real images
    x = np.concatenate((images_train, images_fake))
    y = np.ones([2 * batch_size, 1])
    y[batch_size:, :] = 0

    # Train discriminator for one batch
    d_stats = model_discriminator.train_on_batch(x, y)

    # Train the adversarial model for one batch
    y = np.ones([batch_size, 1])
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    a_stats = model_adversarial.train_on_batch(noise, y)

    if i % 20 == 0:
        print("{}% task complete...".format(round((i / iter) * 100, 2)))

        # Plotting images generated by Generator
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 6))
        noise = np.random.uniform(-1.0, 1.0, size=[40, 100])
        images = net_generator.predict(noise)

        for i in range(40):
            image = images[i, :, :, :]
            image = np.reshape(image, [28, 28])
            plt.subplot(4, 10, i + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
