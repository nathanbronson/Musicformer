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
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(10)
        ])
    
    def encode(self, x):
        return self.encoder(x)
    
    def perturb(self, z, eps=.3):
        return z + eps * tf.random.normal(shape=tf.shape(z))
    
    def decode(self, z):
        return self.decoder(z)
    
    def train_call(self, x):
        a = tf.expand_dims(tf.random.uniform(shape=tf.shape(z)[:-1], minval=-3, maxval=3), axis=-1)
        z = self.encode(x)
        r = tf.range(tf.shape(z)[0])
        sidx, tidx = tf.meshgrid(r, r)
        s = tf.gather(z, sidx)
        t = tf.gather(z, tidx)

        pz = self.perturb(z)
        az = z * a
        stz = s + t

        py = self.decode(pz)
        ay = self.decode(az)
        sty = self.decode(stz)

        pzh = self.encode(py)
        azh = self.encode(ay) / a
        stzh = self.encode(sty)

        all_z = tf.concat([z, az, stz], axis=0)
        all_zh = tf.concat([pzh, azh, stzh], axis=0)

        return all_z, all_zh
    
    def test_call(self, x):
        a = tf.expand_dims(tf.random.uniform(shape=tf.shape(z)[:-1], minval=-3, maxval=3), axis=-1)
        z = self.encode(x)
        r = tf.range(tf.shape(z)[0])
        sidx, tidx = tf.meshgrid(r, r)
        s = tf.gather(z, sidx)
        t = tf.gather(z, tidx)

        pz = z
        az = z * a
        stz = s + t

        py = self.decode(pz)
        ay = self.decode(az)
        sty = self.decode(stz)

        pzh = self.encode(py)
        azh = self.encode(ay) / a
        stzh = self.encode(sty)

        all_z = tf.concat([x, z, az, stz], axis=0)
        all_zh = tf.concat([py, pzh, azh, stzh], axis=0)

        return all_z, all_zh
    
    def train_step(self, data):
        x = data
        
        with tf.GradientTape() as tape:
            y, y_pred = self.train_call(x)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x = data
        
        y, y_pred = self.test_call(x)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}