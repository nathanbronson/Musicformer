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
        self.discriminator = tf.keras.Sequential([#dropout?
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1), name="dc_input"),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3,
                strides=(2, 2), activation='relu'
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3,
                strides=(2, 2), activation='relu'
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
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
        self.discriminator_optimizer = dco
        self.lmse = tf.keras.losses.MeanSquaredError()
        self.lbce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def discriminate(self, x):
        return self.discriminator(x)
    
    def train_call(self, x):
        v = self.encode(x)
        v_p = v + (v * tf.random.normal(tf.shape(v)))
        v_n = tf.reduce_min(v, axis=0) + (tf.random.uniform(tf.shape(v)) * (tf.reduce_max(v, axis=0) - tf.reduce_min(v, axis=0)))

        y = self.decode(v)
        y_p = self.decode(v_p)
        y_n = self.decode(v_n)

        p_x = self.discriminate(x)
        p_y = self.discriminate(tf.math.sigmoid(y))
        p_y_p = self.discriminate(tf.math.sigmoid(y_p))
        p_y_n = self.discriminate(tf.math.sigmoid(y_n))

        return y, p_y, p_x, tf.concat([p_y_p, p_y_n], axis=0)
    
    def test_call(self, x):
        v = self.encode(x)
        v_p = v + (v * tf.random.normal(tf.shape(v)))
        v_n = tf.reduce_min(v, axis=0) + (tf.random.uniform(tf.shape(v)) * (tf.reduce_max(v, axis=0) - tf.reduce_min(v, axis=0)))

        y = self.decode(v)
        y_p = self.decode(v_p)
        y_n = self.decode(v_n)

        p_x = self.discriminate(x)
        p_y = self.discriminate(tf.math.sigmoid(y))
        p_y_p = self.discriminate(tf.math.sigmoid(y_p))
        p_y_n = self.discriminate(tf.math.sigmoid(y_n))

        return y, p_y, p_x, tf.concat([p_y_p, p_y_n], axis=0)
            
    def train_step(self, data):
        x = data
        
        with tf.GradientTape(persistent=True) as tape:
            y, p_y, p_x, p_y_p = self.train_call(x)
            coherence = self.lmse(x, y)
            closure = self.lmse(
                tf.concat([
                    tf.zeros_like(p_y_p, dtype=tf.float32),
                    tf.zeros_like(p_y, dtype=tf.float32)
                ], axis=0),
                tf.math.sigmoid(tf.concat([p_y_p, p_y], axis=0))
            )
            discriminator_loss = self.lmse(
                tf.concat([
                    tf.ones_like(p_y, dtype=tf.float32),
                    tf.zeros_like(p_x, dtype=tf.float32)
                ], axis=0),
                tf.math.sigmoid(tf.concat([p_y, p_x], axis=0))
            )
            encoder_decoder_loss = coherence + closure
        
        encoder_vars = self.encoder.trainable_variables
        decoder_vars = self.decoder.trainable_variables
        discriminator_vars = self.discriminator.trainable_variables

        self.encoder_optimizer.apply_gradients(
            zip(tape.gradient(encoder_decoder_loss, encoder_vars), encoder_vars)
        )
        self.decoder_optimizer.apply_gradients(
            zip(tape.gradient(encoder_decoder_loss, decoder_vars), decoder_vars)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(tape.gradient(discriminator_loss, discriminator_vars), discriminator_vars)
        )
        del tape
        
        self.ed_loss_tracker.update_state(encoder_decoder_loss)
        self.dc_loss_tracker.update_state(discriminator_loss)
        self.coherence_tracker.update_state(coherence)
        self.closure_tracker.update_state(closure)
        
        return {
            "ed_loss": self.ed_loss_tracker.result(),
            "dc_loss": self.dc_loss_tracker.result(),
            "rec": self.coherence_tracker.result(),
            "clo": self.closure_tracker.result()
        }
    
    def test_step(self, data):
        x = data
        
        y, p_y, p_x, p_y_p = self.test_call(x)
        coherence = self.lmse(x, y)
        closure = self.lmse(
            tf.concat([
                tf.zeros_like(p_y_p, dtype=tf.float32),
                tf.zeros_like(p_y, dtype=tf.float32)
            ], axis=0),
            tf.math.sigmoid(tf.concat([p_y_p, p_y], axis=0))
        )
        discriminator_loss = self.lmse(
            tf.concat([
                tf.ones_like(p_y, dtype=tf.float32),
                tf.zeros_like(p_x, dtype=tf.float32)
            ], axis=0),
            tf.math.sigmoid(tf.concat([p_y, p_x], axis=0))
        )
        encoder_decoder_loss = coherence + closure
        
        self.ed_loss_tracker.update_state(encoder_decoder_loss)
        self.dc_loss_tracker.update_state(discriminator_loss)
        self.coherence_tracker.update_state(coherence)
        self.closure_tracker.update_state(closure)
        
        return {
            "ed_loss": self.ed_loss_tracker.result(),
            "dc_loss": self.dc_loss_tracker.result(),
            "rec": self.coherence_tracker.result(),
            "clo": self.closure_tracker.result()
        }