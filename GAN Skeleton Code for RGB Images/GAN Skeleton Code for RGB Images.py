import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Reshape, \
    BatchNormalization, Dropout, Flatten, Dense, Activation, UpSampling2D

# -----------------------------------------------------------------

# Loading all images saved as numpy array
images = np.load(r"Celeba_240x240x3_5kSamples.npy")

for i in range(5):
    plt.imshow(cv2.cvtColor(images[i * 10 + 1], cv2.COLOR_BGR2RGB))
    plt.show()

# Scaling all Images
x_train = images / 255


# -----------------------------------------------------------------

def discriminator():
    net = Sequential()
    input_shape = (240, 240, 3)
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

    net.add(Dense(15 * 15 * 256, input_dim=100))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(Reshape((15, 15, 256)))
    net.add(Dropout(dropout_prob))

    net.add(UpSampling2D(size=(4, 4)))
    net.add(Conv2D(128, kernel_size=(2, 2), strides=2))
    net.add(BatchNormalization())
    net.add(ReLU())

    net.add(UpSampling2D(size=(4, 4)))
    net.add(Conv2D(64, kernel_size=(2, 2), strides=2))
    net.add(BatchNormalization())
    net.add(ReLU())

    net.add(UpSampling2D(size=(4, 4)))
    net.add(Conv2D(32, 2, padding='same', strides=2))
    net.add(BatchNormalization())
    net.add(ReLU())

    net.add(UpSampling2D(size=(4, 4)))
    net.add(Conv2D(16, 2, padding='same', strides=2))
    net.add(BatchNormalization())
    net.add(ReLU())

    net.add(Conv2D(3, 2, padding='same'))
    net.add(Activation('tanh'))

    return net


# -----------------------------------------------------------------

# We create & add the discriminator network to a new Sequential
# model and do not directly compile the discriminator itself.

net_discriminator = discriminator()
optim_discriminator = Adam(learning_rate=0.0002, beta_1=0.5)
model_discriminator = Sequential()
model_discriminator.add(net_discriminator)
model_discriminator.compile(loss='binary_crossentropy', optimizer=optim_discriminator, metrics=['accuracy'])

# -----------------------------------------------------------------

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

# -----------------------------------------------------------------

# Training the DCGAN model
batch_size = 64
iter = 10000

iterations = []
discriminator_losses = []
discriminator_accuracies = []
gan_losses = []
gan_accuracies = []

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

    # ========================================================

    # Records losses & accuracies
    iterations.append(i)
    discriminator_losses.append(d_stats[0])
    discriminator_accuracies.append(d_stats[1])
    gan_losses.append(a_stats[0])
    gan_accuracies.append(a_stats[1])

    # Plots loss & accuracy graphs
    if i % 100 == 0:
        # Keep records of only the last 100 iterations
        # since only the latest metrics matter.
        n = len(iterations) - 100
        iterations = iterations[n:]
        discriminator_losses = discriminator_losses[n:]
        discriminator_accuracies = discriminator_accuracies[n:]
        gan_losses = gan_losses[n:]
        gan_accuracies = gan_accuracies[n:]

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.title("Discriminator Loss")
        plt.ylabel("Loss")
        plt.xlabel("Iterations")
        plt.plot(iterations, discriminator_losses)

        plt.subplot(2, 2, 2)
        plt.title("Discriminator Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Iterations")
        plt.plot(iterations, discriminator_accuracies)

        plt.subplot(2, 2, 3)
        plt.title("GAN Loss")
        plt.ylabel("Loss")
        plt.xlabel("Iterations")
        plt.plot(iterations, gan_losses)

        plt.subplot(2, 2, 4)
        plt.title("GAN Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Iterations")
        plt.plot(iterations, gan_accuracies)

        plt.subplots_adjust(
            left=0.155, bottom=0.117,
            right=0.9, top=0.924,
            wspace=0.2, hspace=0.514)

        plt.show()

    # ========================================================

    # plots images generated by the generator after training.

    if i % 50 == 0:
        print("{}% task complete...".format(round((i / iter) * 100, 2)))

        # Plotting images generated by Generator
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        noise = np.random.uniform(-1.0, 1.0, size=[10, 100])
        images = net_generator.predict(noise)

        for i in range(9):
            image = images[i, :, :, :]
            image = np.reshape(image, [240, 240, 3])
            image = np.clip(image, 0, 1)
            plt.subplot(3, 3, i + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.tight_layout()
        plt.show()