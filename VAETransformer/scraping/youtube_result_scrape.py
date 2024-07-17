import requests
from bs4 import BeautifulSoup

search = 'tree'

headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}

html = requests.get('https://www.youtube.com/results?search_query='+search, headers=headers).text
soup = BeautifulSoup(html, 'html.parser')
for link in soup.find_all('a'):
    if '/watch?v=' in link.get('href'):
      print(link.get('href'))
      # May change when Youtube Website may get updated in the future.
      video_link = link.get('href')