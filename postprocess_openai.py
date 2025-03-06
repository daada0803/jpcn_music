# %%
from openai import OpenAI
from pprint import pprint
import pandas as pd
from pathlib import Path
import shutil

# Initialize the OpenAI API client
client = OpenAI(api_key="sk-")

# %%
def extract_lyrics(singer, song, file_path):
    # Read the text from the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Define the messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"下面文字中记载了歌手{singer}的《{song}》的歌词以及歌词以外的情报。请将歌词提取出来。回答要遵循以下格式。\n歌手 《歌曲》'''\n歌词\n'''\n\n\n{text}\n\n"}
    ]
    # pprint(messages)

    # Call the OpenAI API to process the text
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response

# %%
# lyrics = extract_lyrics("谭咏麟", "酒红色的心", "cn_music/1/cn_lyrics.txt")
# pprint(lyrics.choices[0].message.content.strip())
# with open("tmp.txt", 'w', encoding='utf-8') as f:
#     f.write(lyrics.choices[0].message.content.strip())

# %%
def main():
    songs_650 = pd.read_csv("650songs.csv", index_col=None, header=None, names=["cn_singer", "cn_song", "jp_singer", "jp_song"])
    songs_650_lyrics = pd.read_csv("650songs_lyrics.csv", index_col=None)
    cn_650 = songs_650[["cn_singer", "cn_song"]]
    songs_650_lyrics["cn_lyrics"] = None
    for id_row, content_row in cn_650.iterrows():
        singer = content_row["cn_singer"]
        song = content_row["cn_song"]
        working_dir = Path("cn_music") / str(id_row+1)
        lyrics_file = working_dir / "cn_lyrics.txt"
        if not lyrics_file.exists():
            print(f"File {str(lyrics_file)} does not exist.")
            continue
        print(f"Processing {id_row+1} {singer} {song}")
        try:
            response = extract_lyrics(singer, song, lyrics_file)
            lyrics = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            continue
        if len(lyrics) > 10:
            # shutil.move(str(lyrics_file), working_dir / "cn_lyrics.txt.bak")
            new_lyrics_file = working_dir / "cn_lyrics_processed.txt"
            with open(new_lyrics_file, 'w', encoding='utf-8') as f:
                f.write(lyrics)
            songs_650_lyrics.loc[id_row, "cn_lyrics"] = str(new_lyrics_file)
    songs_650_lyrics.to_csv("650songs_lyrics.csv", index=False)

if __name__ == "__main__":
    main()