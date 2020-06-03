# load mnist data
from keras.datasets.mnist import load_data
(trainX, trainY), (testX, testY) = load_data()
# %%
# plot an example image
import matplotlib.pyplot as plt
plt.imshow(trainX[1], cmap="gray")
print("corresponding digit: "+str(trainY[1]))

# plot multiple images
for i in range(25):
    # define subplot
    plt.subplot(5, 5, 1+i)
    plt.axis('off')
    plt.imshow(trainX[i], cmap='gray_r')
    
plt.show()
# %%
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU
# define discriminator
def define_discriminator(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model

model = define_discriminator()
model.summary()
# %%
from numpy import expand_dims, ones, zeros
from numpy.random import rand, randint

def load_real_samples():
    # load mnist dataset
    (trainX, _), (_, _) = load_data()
    # expand images to 3D (28,28) --> (28,28,1)
    X = expand_dims(trainX, axis=-1)
    # scale images from [0,255] --> [0,1]
    X = X.astype('float32')
    X = X/255.0
    return X

def generate_real_samples(dataset, n_samples):
    # choose random instances --> randint chooses n_samples randomly from the range 0 to dataset.shape[0], which is the number of samples in the dataset
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected imgs
    X = dataset[ix]
    # generate "real" class labels of 1
    y = ones((n_samples, 1))
    return X, y

def generate_fake_samples(n_samples):
    # generate uniform random numbers in  [0,1]
    X = rand(28*28*n_samples) # generates 28*28*n_samples rand nums in total
    # reshape into a batch of grayscale imgs
    X = X.reshape((n_samples,28,28,1))
    # generate 'fake' class labels of 1
    y = zeros((n_samples, 1))
    return X, y

def train_discriminator(model, dataset, n_iter=100, n_batch=256):
    half_batch = int(n_batch/2)
    # manually enumerate for each epoch
    for i in range(n_iter):
        # get the randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # train discriminator on real imgs
        _, real_acc = model.train_on_batch(X_real, y_real)
        # generate 'fake' samples
        X_fake, y_fake = generate_fake_samples(half_batch)
        # train on fake samples
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
        # summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

dataset = load_real_samples()
train_discriminator(model, dataset)

# %%
from keras.models import Sequential
from keras.layers import Reshape, Conv2DTranspose, Dense, Conv2D, LeakyReLU

# takes input from latent space (100d noise vector) and outputs image
def define_generator(latent_dim):
    model = Sequential()
    # foundation for many versions of 7x7 img
    n_nodes = 128*7*7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7,7,128)))
    # upsample from 7x7 to 14x14 --> 128 feature maps, size 14x14
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model


latent_dim = 100
model = define_generator(latent_dim)
model.summary()
# %%
from numpy.random import randn
# generate latent points as input for generator
def generate_latent_points(latent_dim, n_samples):
    # generate points
    x_input = randn(latent_dim*n_samples)
    # reshape so that it can be input
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use generator to generate fake samples and corresponding class labels (0)
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict fake samples
    X = g_model.predict(x_input)
    # create fake class labels (0)
    y = zeros((n_samples,1))
    return X, y

latent_dim = 100
model = define_generator(latent_dim)
n_samples = 25
X, _ = generate_fake_samples(model, latent_dim, n_samples)
# plot the generated samples
for i in range(n_samples):
    # define subplots
    plt.subplot(5,5,1+i)
    plt.axis('off')
    plt.imshow(X[i,:,:,0], cmap="gray_r")
    
plt.show()
# %%
# train the generator
# define combined generator and discriminator model --> used to update the generator
def define_gan(g_model, d_model):
    # make discriminator weights not trainable
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model
# %%
latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)

gan_model.summary()
# %% this will only train the generator via the discriminator's loss, and will leave the discriminator with its default
def train_gan(gan_model, latent_dim, n_epochs=100, n_batch=256):
    for i in range(n_epochs):
        # prepare points in latent space as input for generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for fake samples
        y_gan =  ones((n_batch,1))
        # train generator based on discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)
# %%
from numpy import vstack

# train the generator and the discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset.shape[0]/n_batch)
    half_batch = int(n_batch/2)
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' samples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # create one big array for training the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # prepare ponts in latent space as input for generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for fake samples
            y_gan = ones((n_batch,1))
            # train generator via discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
            # evaluate the model performance, sometimes
            if (i+1) % 10 == 0:
                summarize_performance(i, g_model, d_model, dataset, latent_dim)


# %%
# evaluate discriminator, plot  generated imgs, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real samples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake samples
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    
    #save plot
    save_plot(X_fake, epoch)
    
    # save generator model
    filename = 'models/generator_model_%03d.h5' % (epoch+1)
    g_model.save(filename)

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
    # plot imgs
    for i in range(n*n):
        # define subplot
        plt.subplot(n,n,1+i)
        plt.axis('off')
        plt.imshow(examples[i,:,:,0], cmap="gray_r")
    # save plot to file
    filename = 'plots/generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()
# %%
latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
dataset = load_real_samples()
train(g_model, d_model, gan_model, dataset, latent_dim)
















