# %%
import requests  # For making HTTP requests to APIs and websites
import sys
import os
import subprocess
import json
from time import sleep
from urllib.parse import urlparse
import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError, PostProcessingError
# from spotdl import Spotdl
# from spotdl.utils.config import DEFAULT_CONFIG

# %%
def search(search_item, api_key, cse_id, search_depth=10, site_filter=None):
    service_url = 'https://www.googleapis.com/customsearch/v1'

    params = {
        'q': search_item,
        'key': api_key,
        'cx': cse_id,
        'num': search_depth
    }

    try:
        response = requests.get(service_url, params=params)
        response.raise_for_status()
        results = response.json()

        # Check if 'items' exists in the results
        if 'items' in results:
            if site_filter is not None:
                
                # Filter results to include only those with site_filter in the link
                filtered_results = [result for result in results['items'] if site_filter in result['link']]

                if filtered_results:
                    return filtered_results
                else:
                    print(f"No results with {site_filter} found.")
                    return []
            else:
                if 'items' in results:
                    return results['items']
                else:
                    print("No search results found.")
                    return []

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the search: {e}")
        return []

# %%
def save_webpage(url, output_file, retries=3, delay=1):
    attempts = 0
    # use cookies for baike.baidu.com
    if 'baike.baidu.com' in urlparse(url).netloc:
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        }
        cookies = {'BAIDUID': '217A305C33DE5DB60DF4FADE7A9D3D27:FG=1', 'BDUSS': 'txRk5ZMEhTWEQxWmlSaEppTVVqaVF-NXJTcU5mY1R-ZVRQTVo4LWUxNUVtODFuSVFBQUFBJCQAAAAAAAAAAAEAAADeg6wrytLT0XpyZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEQOpmdEDqZndm'}
        # headers = {
        # 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        # 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        # 'Referer': 'https://www.google.com/search?q=chinese+music',
        # 'Accept-Encoding': 'gzip, deflate, br',
        # 'Accept-Language': 'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4',
        # 'Connection': 'keep-alive',
        # 'Upgrade-Insecure-Requests': '1',
        # 'Host': 'baike.baidu.com',
        # }
        
    else:
        cookies = None
        headers = {'User-Agent': 'Mozilla/5.0'}
    while attempts < retries:
        try:
            response = requests.get(url, headers=headers, cookies=cookies, timeout=10)
            response.raise_for_status()  # Check if the request was successful

            with open(output_file, 'wb') as file:  # Open file in binary write mode
                file.write(response.content)
            print(f"Web page downloaded and saved to {output_file}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            attempts += 1
            sleep(delay)

def download_audio(url, output_filename, output_dir='.', max_retries=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # use cookies for youtube.com and bilibili.com
    if 'youtube.com' in urlparse(url).netloc:
        cookiefile = '/home/zguo/Projects/jpzh_music/www.youtube.com_cookies.txt'
    elif 'bilibili.com' in urlparse(url).netloc:
        cookiefile = '/home/zguo/Projects/jpzh_music/www.bilibili.com_cookies.txt'
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_dir}/{output_filename}.%(ext)s',
        'noplaylist': True,
        'cookiefile': cookiefile,
    }

    attempt = 0
    while attempt < max_retries:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f'Successfully downloaded: {url}')
            return
        except (DownloadError, ExtractorError, PostProcessingError) as e:
            attempt += 1
            print(f'Error downloading {url}: {e}. Attempt {attempt} of {max_retries}.', file=sys.stderr)
            if attempt < max_retries:
                sleep(5)  # Wait for 5 seconds before retrying
            else:
                print(f'Failed to download {url} after {max_retries} attempts.', file=sys.stderr)
        except Exception as e:
            print(f'An unexpected error occurred for {url}: {e}.', file=sys.stderr)
            break

def download_spotify_track(url, output_filename, output_dir='.', max_retries=3):
    """
    Downloads a Spotify track using spotDL with a specified filename and retry mechanism.

    Parameters:
    - url (str): The Spotify track URL.
    - output_filename (str): Desired output filename without extension.
    - output_dir (str): Directory to save the downloaded track.
    - max_retries (int): Maximum number of retry attempts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_template = os.path.join(output_dir, f"{output_filename}") + ".{output-ext}"

    attempt = 0
    while attempt < max_retries:
        try:
            print(f"Attempting to download: {url} (Attempt {attempt + 1} of {max_retries})")
            subprocess.run(
                ['spotdl', url, '--output', output_template],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"Successfully downloaded: {url}")
            return

        except subprocess.CalledProcessError as e:
            attempt += 1
            print(f"Error downloading {url}: {e.stderr}. Attempt {attempt} of {max_retries}.", file=sys.stderr)
            if attempt < max_retries:
                sleep(5)
            else:
                print(f"Failed to download {url} after {max_retries} attempts.", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

# def download_spotify_track(url, output_filename, max_retries=3, output_dir='.'):
#     """
#     Downloads a Spotify track using spotDL with a specified filename and retry mechanism.

#     Parameters:
#     - url (str): The Spotify track URL.
#     - output_filename (str): Desired output filename without extension.
#     - max_retries (int): Maximum number of retry attempts.
#     - output_dir (str): Directory to save the downloaded track.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Update the default configuration with the desired output template
#     config = DEFAULT_CONFIG
#     config['output'] = os.path.join(output_dir, f"{output_filename}.%(ext)s")

#     # Initialize spotDL with the custom configuration
#     spotdl_handler = Spotdl(config)

#     attempt = 0
#     while attempt < max_retries:
#         try:
#             print(f"Attempting to download: {url} (Attempt {attempt + 1} of {max_retries})")
#             # Search for the song using the provided URL
#             songs = spotdl_handler.search([url])
#             if not songs:
#                 raise ValueError(f"No songs found for URL: {url}")

#             # Download the song
#             downloaded_songs = spotdl_handler.download_songs(songs)
#             if downloaded_songs:
#                 print(f"Successfully downloaded: {url}")
#                 return
#             else:
#                 raise Exception(f"Download failed for URL: {url}")

#         except Exception as e:
#             attempt += 1
#             print(f"Error downloading {url}: {e}. Attemp {attempt} of {max_retries}.", file=sys.stderr)
#             if attempt < max_retries:
#                 sleep(5)
#             else:
#                 print(f"Failed to download {url} after {max_retries} attempts.", file=sys.stderr)

# %%
def main():
    cse_id = "92796b2a267ad43f9"
    api_key = "AIzaSyBBgnzo0OuwKq0Tua3P5O0Qkz87QUPny8M"
    # search_term = '谭咏麟 酒红色的心'
    search_term = sys.argv[1]
    
    search_items = search(search_item=search_term, api_key=api_key, cse_id=cse_id, search_depth=10, site_filter=None)
    # save search results to json file
    with open('search_results.json', 'w') as f:
        json.dump(search_items, f)
    
    for item in search_items:
        print(item['title'], file=sys.stderr)
        print(item['link'], file=sys.stderr)
        print(item['snippet'], file=sys.stderr)

    baike_urls = [item for item in search_items if 'baike.baidu.com' in urlparse(item['link']).netloc]
    youtube_urls = [item for item in search_items if 'youtube.com' in urlparse(item['link']).netloc]
    bilibili_urls = [item for item in search_items if 'bilibili.com' in urlparse(item['link']).netloc]
    music163_urls = [item for item in search_items if 'music.163.com' in urlparse(item['link']).netloc]
    apple_urls = [item for item in search_items if 'music.apple.com' in urlparse(item['link']).netloc]
    spotify_urls = [item for item in search_items if 'open.spotify.com' in urlparse(item['link']).netloc]
    qqmusic_urls = [item for item in search_items if 'y.qq.com' in urlparse(item['link']).netloc]
    kugou_urls = [item for item in search_items if 'kugou.com' in urlparse(item['link']).netloc]
    kuwo_urls = [item for item in search_items if 'kuwo.cn' in urlparse(item['link']).netloc]
    migu_urls = [item for item in search_items if 'music.migu.cn' in urlparse(item['link']).netloc]
    xiami_urls = [item for item in search_items if 'xiami.com' in urlparse(item['link']).netloc]
    lizhi_urls = [item for item in search_items if 'lizhi.fm' in urlparse(item['link']).netloc]
    qingting_urls = [item for item in search_items if 'qingting.fm' in urlparse(item['link']).netloc]
    dragonfly_urls = [item for item in search_items if 'dragonfly.fm' in urlparse(item['link']).netloc]
    
    output_dir = 'webpages'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, item in enumerate(baike_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_baidu_baike.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(youtube_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_youtube.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(bilibili_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_bilibili.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(music163_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_music_163.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(apple_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_apple_music.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(spotify_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_spotify.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(qqmusic_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_qq_music.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(kugou_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_kugou.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(kuwo_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_kuwo.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(migu_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_migu.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(xiami_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_xiami.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(lizhi_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_lizhi.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(qingting_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_qingting.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, item in enumerate(dragonfly_urls, start=1):
        url = item.get('link')
        output_file = f'{output_dir}/{idx}_dragonfly.html'
        save_webpage(url, output_file)
        sleep(1)
    
    for idx, item in enumerate(spotify_urls, start=1):
        url = item.get('link')
        output_filename = f'{idx}_spotify'
        download_spotify_track(url, output_filename, output_dir='audio')
        sleep(5)
    for idx, item in enumerate(youtube_urls, start=1):
        url = item.get('link')
        output_filename = f'{idx}_youtube'
        download_audio(url, output_filename, output_dir='audio')
        sleep(5)
    for idx, item in enumerate(bilibili_urls, start=1):
        url = item.get('link')
        output_filename = f'{idx}_bilibili'
        download_audio(url, output_filename, output_dir='audio')
        sleep(5)

    urls = baike_urls + youtube_urls + bilibili_urls + music163_urls + apple_urls + spotify_urls + qqmusic_urls + kugou_urls + kuwo_urls + migu_urls + xiami_urls + lizhi_urls + qingting_urls + dragonfly_urls
    urls = [item['link'] for item in urls]
    with open('urls.txt', 'w') as f:
        for url in urls:
            f.write(f'{url}\n')
    
if __name__ == '__main__':
    main()