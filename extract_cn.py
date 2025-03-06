from bs4 import BeautifulSoup
import json
import pandas as pd
from pathlib import Path

def parse_baidu_baike_basicinfo(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    basic_info = {}

    basic_info_left = soup.find_all("dl", class_=lambda x: x and x.startswith("basicInfoBlock_") and x.endswith("left"))
    basic_info_right = soup.find_all("dl", class_=lambda x: x and x.startswith("basicInfoBlock_") and x.endswith("right"))

    if len(basic_info_left) > 0:
        basic_info_left = basic_info_left[0]
        left_info = basic_info_left.find_all("div", class_=lambda x: x and x.startswith("itemWrapper_"))
        for block in left_info:
            item_names = block.find_all("dt", class_=lambda x: x and 'itemName_' in x)
            item_values = block.find_all("dd", class_=lambda x: x and 'itemValue_' in x)
            for name, value in zip(item_names, item_values):
                key = name.get_text(strip=True).replace('\xa0', '')
                val = value.get_text(strip=True).replace('\xa0', '')
                basic_info[key] = val

    if len(basic_info_right) > 0:
        basic_info_right = basic_info_right[0]
        right_info = basic_info_right.find_all("div", class_=lambda x: x and x.startswith("itemWrapper_"))
        for block in right_info:
            item_names = block.find_all("dt", class_=lambda x: x and 'itemName_' in x)
            item_values = block.find_all("dd", class_=lambda x: x and 'itemValue_' in x)
            for name, value in zip(item_names, item_values):
                key = name.get_text(strip=True).replace('\xa0', '')
                val = value.get_text(strip=True).replace('\xa0', '')
                basic_info[key] = val
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(basic_info, json_file, ensure_ascii=False, indent=4)

    return basic_info

def parse_baidu_baike_lyrics(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    lemma_content_div = soup.find("div", class_="J-lemma-content")
    if lemma_content_div:
        extracted_text = '\n'.join(lemma_content_div.stripped_strings)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        return extracted_text
    else:
        print(f"No <div class='J-lemma-content'> found in {str(file_path)}.")
        return None

def main():
    songs_650 = pd.read_csv("650songs.csv", index_col=None, header=None, names=["cn_singer", "cn_song", "jp_singer", "jp_song"])
    # songs_650_lyrics = pd.read_csv("650songs_lyrics.csv", index_col=None)
    cn_650 = songs_650[["cn_singer", "cn_song"]]
    cn_lyrics = []
    for id_row, content_row in cn_650.iterrows():
        working_dir = Path("cn_music") / str(id_row+1)
        file_path = working_dir / "webpages" / "1_baidu_baike.html"
        basicinfo_json = working_dir / "cn_lyrics.json"
        lyrics_file = working_dir / "cn_lyrics.txt"
        if file_path.exists():
            print(f"Processing {id_row+1} {content_row['cn_singer']} {content_row['cn_song']}")
            parse_baidu_baike_basicinfo(file_path, basicinfo_json)
            lyrics = parse_baidu_baike_lyrics(file_path, lyrics_file)
            if lyrics:
                cn_lyrics.append(str(basicinfo_json))
            else:
                cn_lyrics.append(None)
        else:
            print(f"File {file_path} does not exist")
            cn_lyrics.append(None)
        
    # songs_650_lyrics["cn_lyrics"] = cn_lyrics
    # songs_650_lyrics.to_csv("650songs_lyrics.csv", index=False)

if __name__ == "__main__":
    main()
