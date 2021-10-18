import time
from selenium import webdriver
import chromedriver_binary
from PIL import Image
import io
import os
import requests
import hashlib


sleep_between_interactions = 2
download_num = 3
query = "moss texture"
search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

wd = webdriver.Chrome()
wd.get(search_url.format(q=query))
thumnail_results = wd.find_elements_by_css_selector("img.rg_i")

image_urls = set()
for img in thumnail_results[:download_num]:
    try:
        img.click()
        time.sleep(sleep_between_interactions)
    except Exception:
        continue
    url_candidates = wd.find_elements_by_class_name("Q4LuWd")
    for candidate in url_candidates:
        url = candidate.get_attribute("src")
        if url and 'https' in url:
            image_urls.add(url)

time.sleep(sleep_between_interactions+10)
wd.quit()


image_save_folder_path = 'moss_textures'
for url in image_urls:
    try:
        image_content = requests.get(url).content
    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")
    
    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        file_path = os.path.join(image_save_folder_path, hashlib.sha1(image_content).hexdigest()[:10] + '.png')
        with open(file_path, 'wb') as f:
            image.save(f, "PNG", quality=100)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")