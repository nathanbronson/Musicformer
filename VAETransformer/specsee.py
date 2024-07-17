import matplotlib.pyplot as plt
from glob import glob
from random import choice
from pickle import load

with open(choice(glob("./spectrograms/*.pkl")), "rb") as doc:
    plt.imshow(load(doc))

plt.show()