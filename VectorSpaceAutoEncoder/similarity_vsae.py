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
        self.similarity = tf.keras.Sequential([#dropout?
            tf.keras.layers.InputLayer(input_shape=(28, 28, 2), name="si_input"),
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
        self.si_loss_tracker = tf.keras.metrics.Mean(name="si_loss")
        self.coherence_tracker = tf.keras.metrics.Mean(name="coh")
        self.closure_tracker = tf.keras.metrics.Mean(name="clo")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="rec")
    
    def compile(self, eo, deo, dco, sio):
        super().compile()
        self.encoder_optimizer = eo
        self.decoder_optimizer = deo
        self.discriminator_optimizer = dco
        self.similarity_optimizer = sio
        self.lmse = tf.keras.losses.MeanSquaredError()
        self.lbce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def encode(self, x):
        return self.encoder(x)
    
    def measure_similarity(self, x1, x2):
        return self.similarity(tf.concat([x1, x2], axis=-1))
    
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

        p_y = self.discriminate(y)
        p_x = self.discriminate(x)

        p_yp = self.discriminate(yp)
        p_ay = self.discriminate(ay)

        #s_1 = self.measure_similarity(x, x)
        #s_0 = self.measure_similarity(tf.random.uniform(tf.shape(x), 0, 1), x)
        #s_yx = self.measure_similarity(y, x)

        b_s = tf.shape(v)[0]
        b = tf.range(b_s)
        idx1, idx2 = tf.meshgrid(b, b)
        idx1 = tf.reshape(idx1, (b_s * b_s,))
        idx2 = tf.reshape(idx2, (b_s * b_s,))
        x1 = tf.gather(x, idx1)
        x2 = tf.gather(x, idx2)
        s = self.measure_similarity(x1, x2)
        s_01 = tf.cast(idx1 == idx2, tf.int32)
        s_yx = self.measure_similarity(tf.concat([y, x], axis=0), tf.concat([x, y], axis=0))

        return p_yp, p_ay, y, p_y, p_x, s_01, s, s_yx
    
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

        p_y = self.discriminate(y)
        p_x = self.discriminate(x)

        p_yp = self.discriminate(yp)
        p_ay = self.discriminate(ay)

        #s_1 = self.measure_similarity(x, x)
        #s_0 = self.measure_similarity(tf.random.uniform(tf.shape(x), 0, 1), x)
        #s_yx = self.measure_similarity(y, x)

        b_s = tf.shape(v)[0]
        b = tf.range(b_s)
        idx1, idx2 = tf.meshgrid(b, b)
        idx1 = tf.reshape(idx1, (b_s * b_s,))
        idx2 = tf.reshape(idx2, (b_s * b_s,))
        x1 = tf.gather(x, idx1)
        x2 = tf.gather(x, idx2)
        s = self.measure_similarity(x1, x2)
        s_01 = tf.cast(idx1 == idx2, tf.int32)
        s_yx = self.measure_similarity(tf.concat([y, x], axis=0), tf.concat([x, y], axis=0))

        return p_yp, p_ay, y, p_y, p_x, s_01, s, s_yx
    
    def train_step(self, data):
        x = data
        
        with tf.GradientTape(persistent=True) as tape:
            p_yp, p_ay, y, p_y, p_x, s_01, s, s_yx = self.train_call(x)
            coherence = tf.reduce_mean(self.lbce(tf.ones_like(s_yx), s_yx))
            closure = tf.reduce_mean(self.lbce(
                tf.concat([
                    tf.zeros_like(p_yp, dtype=tf.float32),
                    tf.zeros_like(p_ay, dtype=tf.float32)
                ], axis=0),
                tf.concat([p_yp, p_ay], axis=0)
            ))
            discriminator_loss = tf.reduce_mean(self.lbce(
                tf.concat([
                    tf.ones_like(p_y, dtype=tf.float32),
                    tf.zeros_like(p_x, dtype=tf.float32)
                ], axis=0),
                tf.concat([p_y, p_x], axis=0)
            ))
            similarity_loss = tf.reduce_mean(self.lbce(s_01, s))
            encoder_decoder_loss = (coherence + closure)/2.0
        
        encoder_vars = self.encoder.trainable_variables
        decoder_vars = self.decoder.trainable_variables
        discriminator_vars = self.discriminator.trainable_variables
        similarity_vars = self.similarity.trainable_variables

        self.encoder_optimizer.apply_gradients(
            zip(tape.gradient(encoder_decoder_loss, encoder_vars), encoder_vars)
        )
        self.decoder_optimizer.apply_gradients(
            zip(tape.gradient(encoder_decoder_loss, decoder_vars), decoder_vars)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(tape.gradient(discriminator_loss, discriminator_vars), discriminator_vars)
        )
        self.similarity_optimizer.apply_gradients(
            zip(tape.gradient(similarity_loss, similarity_vars), similarity_vars)
        )
        del tape
        
        self.ed_loss_tracker.update_state(encoder_decoder_loss)
        self.dc_loss_tracker.update_state(discriminator_loss)
        self.si_loss_tracker.update_state(similarity_loss)
        self.reconstruction_tracker.update_state(self.lmse(x, y))
        self.coherence_tracker.update_state(coherence)
        self.closure_tracker.update_state(closure)
        
        return {
            "ed_loss": self.ed_loss_tracker.result(),
            "dc_loss": self.dc_loss_tracker.result(),
            "si_loss": self.si_loss_tracker.result(),
            "coh": self.coherence_tracker.result(),
            "clo": self.closure_tracker.result(),
            "rec": self.reconstruction_tracker.result()
        }
    
    def test_step(self, data):
        x = data
        
        p_yp, p_ay, y, p_y, p_x, s_01, s, s_yx = self.test_call(x)
        coherence = tf.reduce_mean(self.lbce(tf.ones_like(s_yx), s_yx))
        closure = tf.reduce_mean(self.lbce(
            tf.concat([
                tf.zeros_like(p_yp, dtype=tf.float32),
                tf.zeros_like(p_ay, dtype=tf.float32)
            ], axis=0),
            tf.concat([p_yp, p_ay], axis=0)
        ))
        discriminator_loss = tf.reduce_mean(self.lbce(
            tf.concat([
                tf.ones_like(p_y, dtype=tf.float32),
                tf.zeros_like(p_x, dtype=tf.float32)
            ], axis=0),
            tf.concat([p_y, p_x], axis=0)
        ))
        similarity_loss = tf.reduce_mean(self.lbce(s_01, s))
        encoder_decoder_loss = (coherence + closure)/2.0
        
        self.ed_loss_tracker.update_state(encoder_decoder_loss)
        self.dc_loss_tracker.update_state(discriminator_loss)
        self.si_loss_tracker.update_state(similarity_loss)
        self.reconstruction_tracker.update_state(self.lmse(x, y))
        self.coherence_tracker.update_state(coherence)
        self.closure_tracker.update_state(closure)
        
        return {
            "ed_loss": self.ed_loss_tracker.result(),
            "dc_loss": self.dc_loss_tracker.result(),
            "si_loss": self.si_loss_tracker.result(),
            "coh": self.coherence_tracker.result(),
            "clo": self.closure_tracker.result(),
            "rec": self.reconstruction_tracker.result()
        }