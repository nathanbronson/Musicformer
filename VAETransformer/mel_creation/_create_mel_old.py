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
    for n, file in tqdm(list(enumerate(glob("./music/*/*.mp3")))):
        try:
            d = spectro(file)
            with open("./music/spectrograms/mel_{}.pkl".format(str(n)), "wb") as doc:
                dump(d, doc)
        except Exception as err:
            print(file, type(err), err)
            failed.append(file)
    print("\nretrying failures\n")
    for n, file in tqdm(list(enumerate(failed))):
        try:
            d = spectro(file)
            with open("./music/spectrograms/mel_{}.pkl".format(str(n)), "wb") as doc:
                dump(d, doc)
        except Exception as err:
            print(file, type(err), err)
            failed.append(file)
        #with h5py.File('mel_{}.h5'.format(str(n)), 'w') as f:
        #    f.create_dataset('test_data', data=d, dtype=d.dtype)