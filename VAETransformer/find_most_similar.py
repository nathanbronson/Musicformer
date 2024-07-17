from pickle import load
from typing import Dict
from numpy.typing import NDArray
import numpy as np

CATALOG_PATH = "./catalog/z_lookup.pkl"

#options = [
#    "1. Lookup name",
#    "2. Find most similar",
#    "3. Exit"
#]

with open(CATALOG_PATH, "rb") as doc:
    z_lookup: Dict[str, NDArray] = load(doc)

def top_k(song, k=5, square=True):
    q_z = z_lookup[song]
    distances = {name: np.sum(np.power(q_z - z, 2)) for name, z in z_lookup.items()} if square else {name: np.sum(np.abs(q_z - z)) for name, z in z_lookup.items()}
    return sorted(distances.keys(), key=lambda e: distances[e])[:k]

query = input("Enter query:\n").lower()
song_choices = list(filter(lambda e: query in e.lower(), z_lookup.keys()))
for n, i in enumerate(song_choices):
    print("{}: {}".format(str(n + 1), i))
choice = int(input("Choose one:\n")) - 1
print("\n".join(top_k(song_choices[choice], square=True)))

#while True:
#    choice = int(input("Select an option:\n" + "\n".join(options) + "\n"))
#    if choice == 1:
#        query = input("Enter query:\n")
#        song_choices = list(filter(lambda e: query in e, z_lookup.keys()))
#        for n, song_choice in enumerate(song_choices):
#            print("{}: {}".format(str(n + 1), song_choice))
#    elif choice == 2:
#        query = input("Enter query:\n")
#        print("\n".join(filter(lambda e: query in e, z_lookup.keys())))
#    elif choice == 3:
#        exit()