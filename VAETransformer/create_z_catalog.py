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