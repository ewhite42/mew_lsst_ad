import keras 
from keras import layers
import tensorflow as tf

class AE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:

            z = self.encoder(data)

            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum((data - reconstruction)**2, axis=0))

            total_loss = reconstruction_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }
    def test_step(self, data):

        z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum((data - reconstruction)**2, axis=0))


        total_loss = reconstruction_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }

def build_model(hp):
    input_dim = 35 # Feature vector size

    ## how are the min and max values here determined? -MEW
    neurons_first_layer = hp.Int(
      "first_layer",
      min_value=64,
      max_value=128
    )
    
    neurons_second_layer = hp.Int(
      "second_layer",
      min_value=32,
      max_value=64
    )
    neurons_third_layer = hp.Int(
      "third_layer",
      min_value=16,
      max_value=32
    )
    
    ## create instance of Keras tensor with same shape as the data to be input -MEW
    encoder_inputs = keras.Input(shape=(input_dim,))
    
    ## initalize first encoder layer and normalization -MEW
    x = layers.Dense(
       neurons_first_layer
    )(encoder_inputs)
    x = layers.BatchNormalization()(x)
    ## use relu activation function to calculate output of each node based on input -MEW
    x = tf.keras.activations.relu(x, max_value=1, threshold=0.0)
    
    ## initalize second encoder layer and normalization -MEW   
    x = layers.Dense(
        neurons_second_layer
    )(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, max_value=1, threshold=0.0)
    
    ## initalize third encoder layer and normalization -MEW   
    x = layers.Dense(
        neurons_third_layer
    )(x)

    ## initialize latent representation layer -MEW 
    z = layers.Dense(latent_dim, name="z")(x)
    
    ## group encoder layers together into a Model object, which has training / inference features -MEW
    encoder = keras.Model(encoder_inputs, z, name="encoder")

    ## create instance of Keras tensor with same shape as the latent dimension -MEW
    latent_inputs = keras.Input(shape=(latent_dim))

    ## initialize third (innermost) decoder layer -MEW 
    x = layers.Dense(
        neurons_third_layer
    )(latent_inputs)
    x = tf.keras.activations.relu(x, max_value=1, threshold=0.0)
    x = layers.BatchNormalization()(x)
    
    ## initialize second (middle) decoder layer -MEW 
    x = layers.Dense(
        neurons_second_layer
    )(x)
    x = tf.keras.activations.relu(x, max_value=1, threshold=0.0)
    x = layers.BatchNormalization()(x)
    
    ## initialize first (outermost) decoder layer, which uses sigmoid function instead of RELU -MEW 
    x = layers.Dense(
        neurons_first_layer
    )(x)

    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    
    ## group decoder layers together into a Model object, which has training / inference features -MEW
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    ## create instance of AE (autoencoder) object, established with our newly created encoder and decoder models -MEW
    ae = AE(encoder, decoder)

    ## configure the autoencoder model for training -MEW
    ae.compile(optimizer=keras.optimizers.Adam(
        learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4]),
    ), run_eagerly=True)

    return ae

def latent_model(ld):
    global latent_dim
    latent_dim = ld
    return build_model
