import os
import requests
import re

def get_wikimedia_image_url(page_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        print(f"Fetching page: {page_url}")
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()
        
        # Look for the "Original file" link or the main image
        # Pattern: <a href="https://upload.wikimedia.org/wikipedia/commons/..." class="internal"
        # or <div class="fullMedia"><a href="..."
        
        match = re.search(r'href="(https://upload\.wikimedia\.org/wikipedia/commons/[^"]+)" class="internal"', response.text)
        if match:
            return match.group(1)
            
        match = re.search(r'href="(https://upload\.wikimedia\.org/wikipedia/commons/[^"]+)" download=""', response.text)
        if match:
            return match.group(1)
            
        # Fallback to any upload.wikimedia.org link that looks like the file
        match = re.search(r'https://upload\.wikimedia\.org/wikipedia/commons/[a-z0-9]/[a-z0-9]{2}/[^"]+\.(jpg|jpeg|png)', response.text, re.IGNORECASE)
        if match:
            return match.group(0)
            
        return None
    except Exception as e:
        print(f"Error fetching page {page_url}: {e}")
        return None

def download_file(url, filename):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        print(f"Downloading image from: {url}")
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

pages = {
    "project/datasets/input/0_left.jpg": "https://commons.wikimedia.org/wiki/File:Fundus_photograph_of_normal_left_eye.jpg",
    "project/datasets/input/0_right.jpg": "https://commons.wikimedia.org/wiki/File:Fundus_photograph_of_normal_right_eye.jpg",
    "project/datasets/input/1_left.jpg": "https://commons.wikimedia.org/wiki/File:Fundus_photograph-normal_retina_EDA06.JPG"
}

for filename, page_url in pages.items():
    image_url = get_wikimedia_image_url(page_url)
    if image_url:
        download_file(image_url, filename)
    else:
        print(f"Could not find image URL for {page_url}")
