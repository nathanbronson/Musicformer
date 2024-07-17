from glob import glob
from mel_creation.music_utils import process
from tqdm import tqdm
from pickle import dump
from random import sample

MEL_LOOKUP_PATH = "./catalog/mel_lookup.pkl"

files = glob("./music/*/*.mp3") + glob("./music/*/*.m4a") + glob("./music/*/*/*.mp3") + glob("./music/*/*/*.m4a")
files = sample(files, len(files))
database = {}

for file in tqdm(files):
    try:
        database[file.replace("./music/", "").split(".")[0]] = process(file)
    except KeyboardInterrupt as err:
        raise err
    except Exception as err:
        print("failure:", file, type(err), err)

with open(MEL_LOOKUP_PATH, "wb") as doc:
    dump(database, doc)