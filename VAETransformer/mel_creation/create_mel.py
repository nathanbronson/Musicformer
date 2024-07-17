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
import librosa
import numpy as np
from glob import glob
from tqdm import tqdm
#import h5py
from pickle import dump

from music_utils import process as spectro#def spectro(f):
#    y, sr = librosa.load(f)
#    return np.transpose(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr)), (1, 0))

if __name__ == "__main__":
    failed = []
    for file in tqdm(list(glob("./music/*/*.mp3"))):
        n = file.split("/")[-1].split(".")[0]
        try:
            d = spectro(file)
            with open("./music/spectrograms/mel_{}.pkl".format(str(n)), "wb") as doc:
                dump(d, doc)
        except Exception as err:
            print(file, type(err), err)
            failed.append(file)
    print("\nretrying failures\n")
    for file in tqdm(list(failed)):
        n = file.split("/")[-1].split(".")[0]
        try:
            d = spectro(file)
            with open("./music/spectrograms/mel_{}.pkl".format(str(n)), "wb") as doc:
                dump(d, doc)
        except Exception as err:
            print(file, type(err), err)
            failed.append(file)
        #with h5py.File('mel_{}.h5'.format(str(n)), 'w') as f:
        #    f.create_dataset('test_data', data=d, dtype=d.dtype)