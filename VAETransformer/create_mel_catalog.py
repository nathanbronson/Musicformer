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