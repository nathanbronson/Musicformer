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
import numpy as np

from vsae import VSAE

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

if __name__ == "__main__":
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_size = 60000
    batch_size = 1024
    test_size = 10000
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(train_size).batch(batch_size)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(test_size).batch(batch_size)
    )
    
    model = VSAE(2)
    model.compile(
        eo=tf.keras.optimizers.Adam(1e-4),
        deo=tf.keras.optimizers.Adam(1e-4),
        dco=tf.keras.optimizers.Adam(1e-4),
        sio=tf.keras.optimizers.Adam(1e-4)
    )
    tbcb = tf.keras.callbacks.TensorBoard(log_dir="./logs", write_graph=False)
    ckcb = tf.keras.callbacks.ModelCheckpoint("./checkpoints/{epoch:02d}-{val_ed_loss:.2f}-{val_rec:.2f}.h5", save_weights_only=True)
    model.fit(train_dataset, validation_data=test_dataset, epochs=56000, callbacks=[tbcb, ckcb])