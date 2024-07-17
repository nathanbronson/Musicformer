"""
    Musicformer: a neural network for unsupervised embeddings
    Copyright (C) 2024  Nathan Bronson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import tensorflow as tf

class VSAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = tf.keras.Sequential([#dropout??
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1), name="eo_input"),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,), name="de_input"),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2,
                padding='same', activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2,
                padding='same', activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1,
                padding='same'
            )#sigmoid?
        ])
        self.flat = tf.keras.layers.Flatten()
        self.ed_loss_tracker = tf.keras.metrics.Mean(name="ed_loss")
        self.dc_loss_tracker = tf.keras.metrics.Mean(name="dc_loss")
        self.displacement_tracker = tf.keras.metrics.Mean(name="dsp")
        self.coherence_tracker = tf.keras.metrics.Mean(name="coh")
        self.closure_tracker = tf.keras.metrics.Mean(name="clo")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="rec")
    
    def compile(self, eo, deo, dco, sio):
        super().compile()
        self.encoder_optimizer = eo
        self.decoder_optimizer = deo
        self.lmse = tf.keras.losses.MeanSquaredError()
        self.lbce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def train_call(self, x):
        v = self.encode(x)

        y = self.decode(v)

        return y
    
    def test_call(self, x):
        v = self.encode(x)

        y = self.decode(v)

        return y
            
    def train_step(self, data):
        x = data
        
        with tf.GradientTape(persistent=True) as tape:
            y = self.train_call(x)
            coherence = self.lmse(x, y)
            encoder_decoder_loss = coherence
        
        encoder_vars = self.encoder.trainable_variables
        decoder_vars = self.decoder.trainable_variables

        self.encoder_optimizer.apply_gradients(
            zip(tape.gradient(encoder_decoder_loss, encoder_vars), encoder_vars)
        )
        self.decoder_optimizer.apply_gradients(
            zip(tape.gradient(encoder_decoder_loss, decoder_vars), decoder_vars)
        )
        del tape
        
        self.ed_loss_tracker.update_state(encoder_decoder_loss)
        
        return {
            "ed_loss": self.ed_loss_tracker.result()
        }
    
    def test_step(self, data):
        x = data
        
        y = self.test_call(x)
        coherence = self.lmse(x, y)
        encoder_decoder_loss = coherence
        
        self.ed_loss_tracker.update_state(encoder_decoder_loss)
        
        return {
            "ed_loss": self.ed_loss_tracker.result(),
            "rec": 0
        }