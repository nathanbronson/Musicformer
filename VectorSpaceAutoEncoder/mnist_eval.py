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
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from vsae import VSAE
from mnist_test import preprocess_images

TRAIN_INSTEAD_OF_TEST = False

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    if TRAIN_INSTEAD_OF_TEST:
        test_images = train_images
        test_labels = train_labels
    test_images = preprocess_images(test_images)

    x1 = np.expand_dims(test_images[np.random.choice(np.arange(test_images.shape[0]))], 0)
    x2 = np.expand_dims(test_images[np.random.choice(np.arange(test_images.shape[0]))], 0)

    model = VSAE(2)
    model.train_call(x1)
    model.built = True
    ckpts = sorted(glob("./checkpoints/*.h5"), key=lambda x: int(x.split("/")[-1].split("-")[0]))
    file = max([(float(i.split("/")[-1].split("-")[0]), i) for i in ckpts])[1]
    print(file)
    model.load_weights(file)

    v1 = model.encode(x1)
    print(v1)
    y1 = model.decode(v1)
    v2 = model.encode(x2)
    print(v2)
    y2 = model.decode(v2)

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(x1[0])
    axarr[0,1].imshow(y1[0])
    axarr[1,0].imshow(x2[0])
    axarr[1,1].imshow(y2[0])
    plt.show()

    size = 5
    f, axarr = plt.subplots(size,size)
    ivl = np.linspace(-1, 1, size)
    for r in range(size):
        for c in range(size):
            axarr[r,c].imshow(model.decode(np.array([[ivl[r], ivl[c]]]))[0])
    plt.show()

    v = model.encode(test_images)
    plt.scatter(v[:, 0], v[:, 1], c=test_labels)
    plt.show()

    vs = []
    for ckpt in tqdm(ckpts):
        model.load_weights(ckpt)
        vs.append(model.encode(test_images))
    vs = np.array(vs)
    fig, ax = plt.subplots()
    ln = ax.scatter([], [])
    def init():
        ax.set_xlim(np.min(vs[:, :, 0]) - 2, np.max(vs[:, :, 0]) + 2)
        ax.set_ylim(np.min(vs[:, :, 1]) - 2, np.max(vs[:, :, 1]) + 2)
        return ln,
    def update(frame):
        ln = ax.scatter(frame[:, 0], frame[:, 1], c=test_labels)
        return ln,
    ani = FuncAnimation(fig, update, frames=vs, init_func=init, blit=True, interval=25)
    plt.show()