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

    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(
       neurons_first_layer
    )(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, max_value=1, threshold=0.0)
    x = layers.Dense(
        neurons_second_layer
    )(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x, max_value=1, threshold=0.0)
    x = layers.Dense(
        neurons_third_layer
    )(x)

    z = layers.Dense(latent_dim, name="z")(x)
    encoder = keras.Model(encoder_inputs, z, name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim))

    x = layers.Dense(
        neurons_third_layer
    )(latent_inputs)
    x = tf.keras.activations.relu(x, max_value=1, threshold=0.0)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        neurons_second_layer
    )(x)
    x = tf.keras.activations.relu(x, max_value=1, threshold=0.0)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        neurons_first_layer
    )(x)

    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    ae = AE(encoder, decoder)

    ae.compile(optimizer=keras.optimizers.Adam(
        learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4]),
    ), run_eagerly=True)

    return ae

def latent_model(ld):
    global latent_dim
    latent_dim = ld
    return build_model