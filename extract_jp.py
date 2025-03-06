from bs4 import BeautifulSoup
import json
import pandas as pd
from pathlib import Path

def parse_uta_net_html(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    
    # Extract relevant details
    details = {}
    
    # Find lyricist, composer, arranger, and release date using itemprop attribute
    details["lyricist"] = soup.find("a", itemprop="lyricist").get_text(strip=True) if soup.find("a", itemprop="lyricist") else None
    details["composer"] = soup.find("a", itemprop="composer").get_text(strip=True) if soup.find("a", itemprop="composer") else None
    details["arranger"] = soup.find("a", itemprop="arranger").get_text(strip=True) if soup.find("a", itemprop="arranger") else None
    details["release_date"] = soup.find("p", class_="detail").find(string=lambda text: "発売日：" in text).replace("発売日：", "").strip() if soup.find("p", class_="detail") else None

    # Extract lyrics
    lyrics_div = soup.find("div", id="kashi_area", itemprop="text")
    details["lyrics"] = lyrics_div.get_text("\n", strip=True) if lyrics_div else None

    # Write to JSON file
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(details, json_file, ensure_ascii=False, indent=4)
    
    return details
    
def main():
    songs_650 = pd.read_csv("650songs.csv", index_col=None, header=None, names=["cn_singer", "cn_song", "jp_singer", "jp_song"])
    jp_650 = songs_650[["jp_singer", "jp_song"]]
    jp_lyrics = []
    for id_row, content_row in jp_650.iterrows():
        working_dir = Path("jp_lyrics") / str(id_row+1)
        file_path = working_dir / "1_uta-net.html"
        output_path = working_dir / "jp_lyrics.json"
        if file_path.exists():
            print(f"Processing {id_row+1} {content_row['jp_singer']} {content_row['jp_song']}")
            parse_uta_net_html(file_path, output_path)
            jp_lyrics.append(str(output_path))
        else:
            print(f"File {file_path} does not exist")
            # with open(output_path, "w", encoding="utf-8") as json_file:
            #     json.dump({}, json_file, ensure_ascii=False, indent=4)
            jp_lyrics.append(None)
    
    songs_650["jp_lyrics"] = jp_lyrics
    songs_650.to_csv("650songs_lyrics.csv", index=False)
    

if __name__ == "__main__":
    main()