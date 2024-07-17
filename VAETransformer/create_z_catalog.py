"""
    Musicformer: a neural network for unsupervised embeddings
    Copyright (C) 2023  Nathan Bronson

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
from model import make_transformer
from pickle import load, dump
from tqdm import tqdm
from typing import Dict
from numpy.typing import NDArray
import numpy as np
import tensorflow as tf

CHECKPOINT_PATH = "./checkpoints/checkpoint"
MEL_LOOKUP_PATH = "./catalog/mel_lookup.pkl"
Z_LOOKUP_PATH = "./catalog/z_lookup.pkl"

transformer = make_transformer()
transformer.load_weights(CHECKPOINT_PATH)

with open(MEL_LOOKUP_PATH, "rb") as doc:
    mel_lookup: Dict[str, NDArray[np.float32]] = load(doc)

z_lookup = {name: tf.squeeze(transformer.encode(np.expand_dims(np.squeeze(mel), 0))[0]).numpy() for name, mel in tqdm(mel_lookup.items())}

with open(Z_LOOKUP_PATH, "wb") as doc:
    dump(z_lookup, doc)