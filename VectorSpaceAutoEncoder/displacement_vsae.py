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
    def __init__(self, latent_dim, addition_interval=1.0, multiplication_interval=2.0):
        super().__init__()
        self.addition_interval = addition_interval
        self.multiplication_interval = multiplication_interval
        self.encoder = tf.keras.Sequential([
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
            )
        ])
        self.discriminator = tf.keras.Sequential([
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
        self.displacement = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim, latent_dim), name="dp_input"),
            tf.keras.layers.Dense(latent_dim, activation="relu"),
            tf.keras.layers.Dense(latent_dim, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        self.d_x_transform = tf.keras.layers.Lambda(
            lambda x: tf.sqrt(
                tf.reduce_mean(tf.pow(x, 2))
            )
        )
        self.flat = tf.keras.layers.Flatten()
        self.ed_loss_tracker = tf.keras.metrics.Mean(name="ed_loss")
        self.dc_loss_tracker = tf.keras.metrics.Mean(name="dc_loss")
        self.dp_loss_tracker = tf.keras.metrics.Mean(name="dp_loss")
        self.coherence_tracker = tf.keras.metrics.Mean(name="coh")
        self.closure_tracker = tf.keras.metrics.Mean(name="clo")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="rec")
    
    def compile(self, eo, deo, dco, dpo):
        super().compile()
        self.encoder_optimizer = eo
        self.decoder_optimizer = deo
        self.discriminator_optimizer = dco
        self.displacement_optimizer = dpo
        self.lmse = tf.keras.losses.MeanSquaredError()
        self.lbce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def encode(self, x):
        return self.encoder(x)
    
    def measure_displacement(self, x1, x2):
        return self.displacement(tf.concat([
            tf.expand_dims(x1, 1), tf.expand_dims(x2, 1)
        ], axis=1))
    
    def decode(self, z):
        return self.decoder(z)
    
    def discriminate(self, x):
        return self.discriminator(x)
    
    def train_call(self, x):
        v = self.encode(x)

        y = self.decode(v)

        vp = v + tf.random.uniform(
            tf.shape(v), -self.addition_interval, self.addition_interval
        )
        av = v * tf.expand_dims(tf.random.uniform(
            tf.shape(v)[:-1], -self.multiplication_interval, self.multiplication_interval
        ), -1)

        yp = self.decode(vp)
        ay = self.decode(av)

        v_h = self.encode(y)
        p_y = self.discriminate(y)
        p_x = self.discriminate(x)

        p_yp = self.discriminate(yp)
        p_ay = self.discriminate(ay)

        b_s = tf.shape(v)[0]
        b = tf.range(b_s)
        v1idx, v2idx = tf.meshgrid(b, b)
        v1idx = tf.reshape(v1idx, (b_s * b_s,))
        v2idx = tf.reshape(v2idx, (b_s * b_s,))
        v1 = tf.gather(v, v1idx)
        v2 = tf.gather(v, v2idx)
        x1 = tf.gather(x, v1idx)
        x2 = tf.gather(x, v2idx)

        d_v = self.measure_displacement(v1, v2)
        d_x = self.d_x_transform(x1 - x2)

        return v, v_h, p_yp, p_ay, d_v, d_x, y, p_y, p_x
    
    def test_call(self, x):
        v = self.encode(x)

        y = self.decode(v)

        vp = v + tf.random.uniform(
            tf.shape(v), -self.addition_interval, self.addition_interval
        )
        av = v * tf.expand_dims(tf.random.uniform(
            tf.shape(v)[:-1], -self.multiplication_interval, self.multiplication_interval
        ), -1)

        yp = self.decode(vp)
        ay = self.decode(av)

        v_h = self.encode(y)
        p_y = self.discriminate(y)
        p_x = self.discriminate(x)

        p_yp = self.discriminate(yp)
        p_ay = self.discriminate(ay)

        b_s = tf.shape(v)[0]
        b = tf.range(b_s)
        v1idx, v2idx = tf.meshgrid(b, b)
        v1idx = tf.reshape(v1idx, (b_s * b_s,))
        v2idx = tf.reshape(v2idx, (b_s * b_s,))
        v1 = tf.gather(v, v1idx)
        v2 = tf.gather(v, v2idx)
        x1 = tf.gather(x, v1idx)
        x2 = tf.gather(x, v2idx)

        d_v = self.measure_displacement(v1, v2)
        d_x = self.d_x_transform(x1 - x2)

        return v, v_h, p_yp, p_ay, d_v, d_x, y, p_y, p_x
    
    def train_step(self, data):
        x = data
        
        with tf.GradientTape(persistent=True) as tape:
            v, v_h, p_yp, p_ay, d_v, d_x, y, p_y, p_x = self.train_call(x)
            coherence = (tf.reduce_mean(self.lmse(v, v_h)) + tf.reduce_mean(self.lmse(x, y)))/2.0
            closure = tf.pow((tf.math.sigmoid(p_yp) + tf.math.sigmoid(p_ay))/2.0, 2)
            discriminator_loss = tf.reduce_mean(self.lbce(
                tf.concat([
                    tf.ones_like(p_y, dtype=tf.float32),
                    tf.zeros_like(p_x, dtype=tf.float32)
                ], axis=0),
                tf.concat([p_y, p_x], axis=0)
            ))
            displacement_loss = tf.reduce_mean(self.lmse(d_x, d_v))
            encoder_decoder_loss = (coherence + closure + displacement_loss)/3.0
        
        encoder_vars = self.encoder.trainable_variables
        decoder_vars = self.decoder.trainable_variables
        discriminator_vars = self.discriminator.trainable_variables
        displacement_vars = self.displacement.trainable_variables

        self.encoder_optimizer.apply_gradients(
            zip(tape.gradient(encoder_decoder_loss, encoder_vars), encoder_vars)
        )
        self.decoder_optimizer.apply_gradients(
            zip(tape.gradient(encoder_decoder_loss, decoder_vars), decoder_vars)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(tape.gradient(discriminator_loss, discriminator_vars), discriminator_vars)
        )
        self.displacement_optimizer.apply_gradients(
            zip(tape.gradient(displacement_loss, displacement_vars), displacement_vars)
        )
        del tape
        
        self.ed_loss_tracker.update_state(encoder_decoder_loss)
        self.dc_loss_tracker.update_state(discriminator_loss)
        self.dp_loss_tracker.update_state(displacement_loss)
        self.reconstruction_tracker.update_state(self.lmse(x, y))
        self.coherence_tracker.update_state(coherence)
        self.closure_tracker.update_state(closure)
        
        return {
            "ed_loss": self.ed_loss_tracker.result(),
            "dc_loss": self.dc_loss_tracker.result(),
            "dp_loss": self.dp_loss_tracker.result(),
            "coh": self.coherence_tracker.result(),
            "clo": self.closure_tracker.result(),
            "rec": self.reconstruction_tracker.result()
        }
    
    def test_step(self, data):
        x = data
        
        v, v_h, p_yp, p_ay, d_v, d_x, y, p_y, p_x = self.train_call(x)
        coherence = (tf.reduce_mean(self.lmse(v, v_h)) + tf.reduce_mean(self.lmse(x, y)))/2.0
        closure = tf.pow((tf.math.sigmoid(p_yp) + tf.math.sigmoid(p_ay))/2.0, 2)#(tf.reduce_mean(self.lbce(tf.zeros_like(p_yp), p_yp)) + tf.reduce_mean(self.lbce(tf.zeros_like(p_ay), p_ay)))/2.0
        discriminator_loss = tf.reduce_mean(self.lbce(
            tf.concat([
                tf.ones_like(p_y, dtype=tf.float32),
                tf.zeros_like(p_x, dtype=tf.float32)
            ], axis=0),
            tf.concat([p_y, p_x], axis=0)
        ))
        displacement_loss = tf.reduce_mean(self.lmse(d_x, d_v))
        encoder_decoder_loss = (coherence + closure + displacement_loss)/3.0
        
        self.ed_loss_tracker.update_state(encoder_decoder_loss)
        self.dc_loss_tracker.update_state(discriminator_loss)
        self.dp_loss_tracker.update_state(displacement_loss)
        self.reconstruction_tracker.update_state(self.lmse(x, y))
        self.coherence_tracker.update_state(coherence)
        self.closure_tracker.update_state(closure)
        
        return {
            "ed_loss": self.ed_loss_tracker.result(),
            "dc_loss": self.dc_loss_tracker.result(),
            "dp_loss": self.dp_loss_tracker.result(),
            "coh": self.coherence_tracker.result(),
            "clo": self.closure_tracker.result(),
            "rec": self.reconstruction_tracker.result()
        }