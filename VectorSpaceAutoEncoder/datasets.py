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
import numpy as np
from functools import partial

class VectorSpaceDataset:
    def __init__(self, num_examples, dimension, gamma_inverse):
        self.coordinates = np.random.uniform(0, 1, size=(num_examples, dimension))
        self.x = gamma_inverse(self.coordinates)

def nullify_coords(coords, null_dims=1, null_val=0):
    n_coords = np.shape(coords)[1:2]
    coords[:, np.random.choice(np.arange(n_coords), size=null_dims)] = null_val
    return coords

class SomeNullity:
    """
    used to test whether it can compress away some simple nullity (test with input_dim=dimension and latent_dim=(dimensions-null_dims))
    """
    def __init__(self, num_examples, dimension, null_dims=1, null_val=0):
        self._ds = VectorSpaceDataset(num_examples, dimension, partial(nullify_coords, null_dims=null_dims, null_val=null_val))
        self.x = self._ds.x

class SurjectiveAddition:
    """
    used to test wheter it can correct a simple bijective nonlinearity
    """
    def __init__(self, num_examples, dimension, add_val=1):
        self._ds = VectorSpaceDataset(num_examples, dimension, lambda x: x + add_val)
        self.x = self._ds.x

class NullityAddition:
    """
    used to test compound of nullity and simple nonlinearity (test with input_dim=dimension and latent_dim=(dimensions-null_dims))
    """
    def __init__(self, num_examples, dimension, add_val=1, null_val=1, null_dims=1):
        self._ds = VectorSpaceDataset(num_examples, dimension, lambda x: nullify_coords(x + add_val, null_dims=null_dims, null_val=null_val))
        self.x = self._ds.x

class RToR2:
    """
    used to test simple dimension reduction (test with input_dim=2 and latent_dim=1)
    """
    def __init__(self, num_examples):
        self._ds = VectorSpaceDataset(num_examples, 2, lambda x: np.sum(x, axis=-1, keepdims=True))
        self.x = self._ds.x

class :
    """
    find a way to losslessly represent 
    """
    def __init__(self, num_examples):