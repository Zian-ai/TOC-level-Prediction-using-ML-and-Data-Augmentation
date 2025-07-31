print("Let's begin!")
from keras.layers import Input, Dense, Flatten, Dropout, Reshape
from keras.layers import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils import Progbar
from keras.initializers import RandomNormal
import keras.backend as K
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy as np
import pandas as pd
from keras.optimizers.legacy import RMSprop

# NB_EPOCHS = 20
# D_ITERS = 5
# BATCH_SIZE = 128
# BASE_N_COUNT = 128
# LATENT_SIZE = 100
# SAMPLE_SIZE = 8192

# THIS PROCESS CSV FUNCTION IS ONLY USED WHEN RUNNING ENGINGE.PY ALONE AND IS NOT REFERENCED IN COLUMNS.PY
def process_csv(csv_path):
    # Input: The path location of the CSV  
    # Outputs:
    #       1. The CSV scaled down to be between -1 and 1
    #       2. An array of maximum absolute values for each column.

     real_full_data = pd.read_csv(csv_path, header=0)
     real_full_data = real_full_data.dropna()

    # Select only numeric columns
     numeric_data = real_full_data.select_dtypes(include=[np.number])

    # Store the maximum absolute value of each column
     col_max_array = numeric_data.abs().max().to_frame() 

    # Scale the data to be between -1 and 1
     real_scaled_data = numeric_data / numeric_data.abs().max()

     return real_scaled_data, col_max_array, real_full_data


 
class GAN():
    def __init__(self, real_scaled_data, col_max_array):
        self.real_data = real_scaled_data                            #Stores the real data
        self.col_max_array = pd.DataFrame(col_max_array).T           #Convert to make an easier operations 
        self.data_dim = len(real_scaled_data.columns)                #Numbers of Features 
        self.latent_dim = 100                                        #Noise 
        self.base_n_count = 256                                      # Numbers of Base Neurons
        self.d_iters = 5                                             # Number of Iterations

        # optimizer = RMSprop(lr=0.00005)                              #Minimize the loss function/ Adjust the weights
        optimizer = RMSprop(learning_rate=0.00005)

        self.discriminator = self.build_discriminator()     
        self.discriminator.compile(
            loss=wasserstein_loss,  # Use penalized loss here
            optimizer=optimizer,
        )

        self.generator = self.build_generator()                     #Archictecture of the generator

        z = Input(shape=(self.latent_dim,), name='input_z')         #Representing the random noise vector
        generated_data = self.generator(z)                          # Passes the random noise vector (z) through the generator to produce synthetic data.

        is_fake = self.discriminator(generated_data)                #Evaluate whether it is real or fake.

        self.combined = Model(z, is_fake)                           #Combines the generator and discriminator into a single combined model.
        self.combined.get_layer('D').trainable = False              #self.combined.get_layer('D').trainable = False
        self.combined.compile(loss=wasserstein_loss, optimizer=optimizer)   #This loss function guides the generator to produce data that the discriminator classifies as real.

    def build_generator(self):                                      #responsible for producing "fake" data that mimics the real data distribution.

        weight_init = RandomNormal(mean=0., stddev=0.02)            #Ensures weights start with small, random values, which can stabilize training.
        model = Sequential()                                        #Layers are added one after another

        model.add(Dense(self.base_n_count, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.base_n_count * 2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.base_n_count * 4))
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(Dense(self.base_n_count * 6))                     #Increases the model's capacity to learn complex transformations.    
        model.add(LeakyReLU(alpha=0.2))                                                                                            
        model.add(Dense(self.data_dim, activation='tanh'))          #Outputs synthetic data with dimensions matching the real dataset 
                                                                    #Scales the output to match the normalized range of real data ([-1, 1]).
        noise = Input(shape=(self.latent_dim,))
        fake_data = model(noise)

        print(model.summary())
        # plot_model(model, to_file='generator.png', show_shapes=True, show_layer_names=True)   #Helps document the model’s architecture visually, useful in research or presentations.    OPTIONAL   

        return Model(noise, fake_data, name='G')                     #It creates and returns a Keras Model object that defines the generator in the Generative Adversarial Network (GAN)

    def build_discriminator(self):                          

        weight_init = RandomNormal(mean=0., stddev=0.02)
        model = Sequential()

        model.add(Dense(self.base_n_count * 4, input_dim=self.data_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(Dense(self.base_n_count * 2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(Dense(self.base_n_count))                      
        model.add(LeakyReLU(alpha=0.2))                          
        model.add(Dropout(0.3))                                                          
        model.add(Dense(1, activation='linear'))

        print(model.summary())
        # plot_model(model, to_file='discriminator.png', show_shapes=True, show_layer_names=True)

        data_features = Input(shape=(self.data_dim,))
        is_fake = model(data_features)

        return Model(data_features, is_fake, name='D')

# combine GAN 훈련
    def train(self, epochs=1500, batch_size=13, sample_size=14):  # Observation/historical data    #Epochs (100, 200, 300, 500, 1000, 1500, 3000)
        self.fake_data_list = []                                                            #Stores generated (fake) data at specific intervals during training.
        self.epoch_gen_loss = []                                                            #Tracks the generator's loss after each batch update.
        self.epoch_disc_true_loss = []                                                      #Tracks the discriminator's loss on real data during each batch update.
        self.epoch_disc_fake_loss = []                                                      #Tracks the discriminator's loss on fake data during each batch update.
        nb = int(sample_size / batch_size) * epochs                                         #Calculates the number of batches required to process the sample_size once.
        rounds = int(sample_size / batch_size)                                              #Indicates how many times the model will update in a single epoch.

        progress_bar = Progbar(target=nb)                                                   #A utility to display a progress bar in the console, providing real-time feedback on the training progress.

        for index in range(nb):                                                             #Index Represents the current iteration of the loop
            x_train = self.real_data.sample(sample_size)                                    #Contains the real dataset, scaled between -1 and 1.  Sampling ensures that each batch is different, promoting generalization.
            progress_bar.update(index)                                                      #Sampling ensures that each batch is different, promoting generalization.
            for d_it in range(self.d_iters):                                                #The discriminator is trained self.d_iters times before each generator update
                # unfreeze D                                                                #The discriminator's weights are frozen during the generator's training phase to ensure that only the generator is updated
                self.discriminator.trainable = True                                         #Before training the discriminator, its weights need to be unfrozen, so the model and its layers are made trainable.       
                for l in self.discriminator.layers:                                         #for l in self.discriminator.layers: Iterates through all the layers of the discriminator and ensures each layer is trainable.
                    l.trainable = True

                # clip D weights 
                for l in self.discriminator.layers:                                         # discriminator (critic) needs to satisfy the Lipschitz constraint, which ensures stable training. 
                    weights = l.get_weights()                                               #Retrieves the current weights of the layer.
                    weights = [np.clip(w, -0.01, 0.01) for w in weights]                    #Clips each weight value to be within the range
                    l.set_weights(weights)                                                  # Updates the layer's weights with the clipped values.

                # Maximize D output on reals == minimize -1*(D(real)) and get a batch of real data           #The discriminator is trained to maximize its output for real data.
                data_index = np.random.choice(len(x_train), batch_size, replace=True)  # False               #The discriminator processes a randomly selected batch of real data.
                data_batch = x_train.values[data_index]                                                      #Converts the pandas DataFrame x_train into a NumPy array for efficient numerical operations.

                self.epoch_disc_true_loss.append(self.discriminator.train_on_batch(data_batch, -np.ones(batch_size)))   #Trains the discriminator on a batch of real data and appends the resulting loss to a tracking list

                # Minimize D output on fakes                                                                #Create random noise vectors that the generator can use to produce fake data sample
                # generate a new batch of noise                                                             #
                noise = np.random.normal(loc=0.0, scale=1, size=(int(batch_size), self.latent_dim))

                generated_data = self.generator.predict(noise, verbose=0)                                   #The generator model (self.generator) takes the noise as input and transforms it into synthetic (fake) data.
                self.epoch_disc_fake_loss.append(                                                           
                    self.discriminator.train_on_batch(generated_data, np.ones(int(batch_size))))           
            # freeze D and C                                                                                
            self.discriminator.trainable = False                                                            
            for l in self.discriminator.layers:                                                            
                l.trainable = False

            noise = np.random.normal(loc=0.0, scale=1, size=(int(batch_size), self.latent_dim))             
            self.epoch_gen_loss.append(self.combined.train_on_batch(noise, -np.ones(int(batch_size))))     
                                                                                                            
        

            if (index % int(nb / 5) == 0):                                                                  
                self.fake_data_list.append(self.gen_fake_data(350))  #Number of Synthetic Data to be generated (1:3, 1:5, 1:10)


    def gen_fake_data(self, N=350):  # Number of Synthetic Data to be generated (1:3, 1:5, 1:10)                                                             
        # Uses generator to generate fake data.                                                         

        fake_data = pd.DataFrame()                                                                         

        for x in range(N):                                                                                 
            noise = np.random.normal(0, 1, (1, self.latent_dim))                                           
            gen_data = self.generator.predict(noise)                                                       
            gen_data = gen_data * self.col_max_array                                                       
            fake_data = pd.concat([fake_data, pd.DataFrame(gen_data)], ignore_index=True)                   

        return fake_data                                                                                   


def wasserstein_loss(y_true, y_pred):                                                                       
    # Returns the result of the wasserstein loss funCan you explaiction.
    return K.mean(y_true * y_pred)                                                                          
                                                                                                            
# ALL CODE BELOW IS ONLY USED WHEN RUNNING ENGINE.PY ALONE AND IS NOT USED IN COLUMNS.PY
# THIS CODE'S ONLY PURPOSE IS FOR TROUBLESHOOTING
# __________________________________________________________________________________________________________________________________________

#    # Show generator loss value over iteration epochs.
def show_gen_loss(gan):                                                                                     
#    loss_plot = plt.figure(1)                                                                             
    fig, ax = plt.subplots(1,2,figsize=(20,10))

    ax[0] = plt.subplot(1,2,1)
    ax[0].plot(gan.epoch_gen_loss, color='red', label='generator')
    plt.legend(loc='upper right', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    ax[1] = plt.subplot(1,2,2)
    ax[1].plot(gan.epoch_disc_fake_loss, color='green', label='D_real')
    ax[1].plot(gan.epoch_disc_true_loss, color='blue', label='D_fake')
    plt.legend(loc='upper right', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    ax[0].set_title('Generator loss', fontsize=20)
    ax[0].set_xlabel('Iteration', fontsize=20)
    ax[0].set_ylabel('Loss', fontsize=20)

    ax[1].set_title('Discriminator loss', fontsize=20)
    ax[1].set_xlabel('Iteration', fontsize=20)
    ax[1].set_ylabel('Loss', fontsize=20)

    plt.tight_layout()
    plt.show()

def real_vs_fake(real_data, fake_data_list, x_col, y_col):                                              #Visually compares real data and fake data generated by the GAN.
    fig, axs = plt.subplots(5, sharex=False, figsize=(10,10))
    plt.suptitle('Real and Fake Data Comparison', fontsize=20)

    colors = ('blue', 'red')
    groups = ('fake', 'real')

    for pnum in range(len(fake_data_list)):                                                             #loop variable that represents the current index  pnum: plot number, leg-lenght
        fake_data = fake_data_list[pnum]                                                                #Retrieves the fake data snapshot at index pnum.
        fake_data_points = (fake_data[x_col], fake_data[y_col])                                         #Extracts the values of x_col and y_col from both the fake and real datasets.
        real_data_points = (real_data[x_col], real_data[y_col])                 
        total_data = (fake_data_points, real_data_points)                                               #Combines the real and fake data points for easier iteration or plotting.

        for total_data, color, group in zip(total_data, colors, groups):                                
            x, y = total_data
            axs[pnum].scatter(x, y, alpha=0.2, c=color, edgecolors='none', s=30, label=group)
            # axs[pnum].set_yticks([20, 40, 60, 80, 100])                                                #Y-axis format

    plt.xlabel(x_col, fontsize=15)
    plt.ylabel(y_col, fontsize=15)
    plt.legend(loc=2, fontsize=9)                                                                      #loc: location of the legend
    plt.savefig(r'C:\Users\Christian\Downloads\Predicting_TOC\GAN-TOC\RESULTS\1500-EPOCHS\2-도천-TOC_GAN_data_Epochs(1500)_(300-SYN).png'.format(x_col),dpi=300)  #Save the results

def export_fakes(gan, destination, N=350):  # Number of Synthetic Data to be generated (1:3, 1:5, 1:10)
    list_of_fakes = gan.gen_fake_data(N)
    export_csv = list_of_fakes.to_csv(destination)
    return export_csv

if __name__ == '__main__':
     cc_scaled_data, col_max, cc_data = process_csv(r'C:\Users\Christian\Downloads\Predicting_TOC\CHECKING -2\2-도천\1-INPUT DATA\2-SACHEON_TOC_INPUT_MONTHLY\Quick_test_data.csv') #Files and its directories
     gan = GAN(cc_scaled_data, col_max)
     gan.train(epochs=1500, batch_size=13, sample_size=14)  #sample_size (Observation/historical data)    #Epochs (100, 200, 300, 500, 1000, 1500, 3000)
     export_fakes(gan, r'C:\Users\Christian\Downloads\Predicting_TOC\GAN-TOC\RESULTS\1500-EPOCHS\2-도천-TOC_GAN_data_Epochs(1500)_(300-SYN).csv', 350)  # Number of Synthetic Data to be generated (1:3, 1:5, 1:10)
     
     show_gen_loss(gan)                                                   
     cc_sample_data = cc_data.sample(14)  # sample (Observation/historical data)
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'prcp', 'TOC')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmax', 'TOC')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmin', 'TOC')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'rhum', 'TOC')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'month', 'TOC') 
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'TOC', 'prcp')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmax', 'prcp') 
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmin', 'prcp')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'rhum', 'prcp')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'month', 'prcp')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'TOC', 'tmax')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'prcp', 'tmax') 
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmin', 'tmax')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'rhum', 'tmax')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'month', 'tmax')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'TOC', 'tmin')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'prcp', 'tmin') 
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmax', 'tmin')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'rhum', 'tmin')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'month', 'tmin') 
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'TOC', 'rhum')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'prcp', 'rhum') 
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmax', 'rhum')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmin', 'rhum')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'month', 'rhum') 
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'TOC', 'month')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'prcp', 'month')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmin', 'month')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'tmax', 'month')  
     real_vs_fake(cc_sample_data, gan.fake_data_list, 'rhum', 'month')  
     
print("Successfully Created a Synthetic Data")




