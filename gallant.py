import requests
import random
import shutil
import bs4
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def search_image(word):
    res = requests.get("https://www.google.com/search?hl=jp&q=" + word + "&btnG=Google+Search&tbs=0&safe=off&tbm=isch")
    html = res.text
    soup = bs4.BeautifulSoup(html, 'lxml')
    links = soup.find_all("img")
    link = random.choice(links).get("src")
    return link


def download_image(url, filename):
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open(filename + ".png", "wb") as f:
            res.raw.decode_content = True
            shutil.copyfileobj(res.raw, f)


num = input("検索回数:")
word = input("検索ワード:")
for i in range(int(num)):
    link = search_image(word)
    download_image(link, str(i))
print("OK")