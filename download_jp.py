# %%
import requests  # For making HTTP requests to APIs and websites
import sys
import json
from time import sleep
from urllib.parse import urlparse

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
    while attempts < retries:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Check if the request was successful

            with open(output_file, 'wb') as file:  # Open file in binary write mode
                file.write(response.content)
            print(f"Web page downloaded and saved to {output_file}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            attempts += 1
            sleep(delay)

# %%
def main():
    cse_id = "a1d1da029b71840bf"
    api_key = "AI"
    # search_term = '福山雅治 桜坂'
    search_term = sys.argv[1]
    
    search_items = search(search_item=search_term, api_key=api_key, cse_id=cse_id, search_depth=10, site_filter=None)
    # save search results to json file
    with open('search_results.json', 'w') as f:
        json.dump(search_items, f)
    
    for item in search_items:
        print(item['title'], file=sys.stderr)
        print(item['link'], file=sys.stderr)
        print(item['snippet'], file=sys.stderr)
    
    uta_net_urls = [item['link'] for item in search_items if 'uta-net' in urlparse(item['link']).netloc]
    joysound_urls = [item['link'] for item in search_items if 'joysound' in urlparse(item['link']).netloc]
    else_urls = [item['link'] for item in search_items if 'uta-net' not in urlparse(item['link']).netloc and 'joysound' not in urlparse(item['link']).netloc]
    
    for idx, url in enumerate(uta_net_urls, start=1):
        output_file = f'{idx}_uta-net.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, url in enumerate(joysound_urls, start=1):
        output_file = f'{idx}_joysound.html'
        save_webpage(url, output_file)
        sleep(1)
    for idx, url in enumerate(else_urls, start=1):
        output_file = f'{idx}_others.html'
        save_webpage(url, output_file)
        sleep(1)

if __name__ == '__main__':
    main()
