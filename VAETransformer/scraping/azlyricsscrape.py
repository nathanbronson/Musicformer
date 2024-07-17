from util import stealth_get
import re
from pickle import dump
from tqdm import tqdm
from itertools import chain
from time import sleep

CACHE_DIR = "./data/cache/"
ALPHABET_PATH = "alphabet_links.pkl"
ARTIST_PATH = "artist_songs.pkl"
LYRICS_PATH = "artist_song_lyrics.pkl"
LYRICS_TOO = False
VERBOSE = True
TIMEOUT = 3

#agent = 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) \
#        Gecko/20100101 Firefox/24.0'
#headers = {'User-Agent': agent}#'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
artist_alphabet_url = "https://www.azlyrics.com/{}.html"
az_base_url = "https://www.azlyrics.com/"
alphabet = [i for i in "abcdefghijklmnopqrstuvwxyz"] + ["19"]
alphabet_links = {}
alpha_pattern = re.compile(r"<a href=\"(.*?)\">(.*?)<\/a>")
song_pattern = re.compile(r"<a href=\"(.*?)\".*?>(.*?)<\/a>")
lyrics_pattern = re.compile(r"<b>\".*?\"<\/b>(?:.|\n)*?<div>(?:(?:.|\n)*?<\!(?:.|\n)*?>)?((?:.|\n)*?)<\/div>")
artist_songs = {}
artist_song_lyrics = {}

for alpha in tqdm(alphabet):
    html = stealth_get(artist_alphabet_url.format(alpha), timeout=TIMEOUT).text
    with open("./test_html.txt", "w") as doc:
        doc.write(html)
    artists = [{"link": match.group(1), "artist": match.group(2)} for match in alpha_pattern.finditer(html)]#(link, artist_name)
    alphabet_links[alpha] = artists
    sleep(.1)
if VERBOSE:
    try:
        print(len(alphabet_links.items()), end="  ")
        print(min([len(i) for i in alphabet_links.values()]), end="  ")
        print(max([len(i) for i in alphabet_links.values()]), end="  ")
        print()
    except Exception as err:
        print(type(err), err)
with open(CACHE_DIR + ALPHABET_PATH, "wb") as doc:
    dump(alphabet_links, doc)
for artist_dict in tqdm(list(chain(alphabet_links.values()))):
    html = stealth_get(az_base_url + artist_dict["link"], timeout=TIMEOUT).text
    artist_songs[artist_dict["artist"]] = [{"link": match.group(1), "song": match.group(2)} for match in song_pattern.finditer(html)]
if VERBOSE:
    try:
        print(len(artist_songs.items()), end="  ")
        print(min([len(i) for i in artist_songs.values()]), end="  ")
        print(max([len(i) for i in artist_songs.values()]), end="  ")
        print()
    except Exception as err:
        print(type(err), err)
with open(CACHE_DIR + ARTIST_PATH, "wb") as doc:
    dump(artist_songs, doc)
if LYRICS_TOO:
    for artist, songs in tqdm(list(artist_songs.items())):
        song_lyrics = {}
        for song in songs:
            html = stealth_get(az_base_url + song["link"], timeout=TIMEOUT).text
            try:
                song_lyrics[song["song"]] = lyrics_pattern.search(lyrics_pattern).group(1).replace("<br>", "").replace("<i>", "").replace("</i>", "")
            except KeyboardInterrupt as err:
                raise err
            except Exception as err:
                print(artist, song["song"], type(err), err)
        artist_song_lyrics[artist] = song_lyrics
    with open(CACHE_DIR + LYRICS_PATH, "wb") as doc:
        dump(artist_song_lyrics, doc)