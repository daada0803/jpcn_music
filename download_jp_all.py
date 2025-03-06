# %%
import sys
import pandas as pd
from pathlib import Path
import subprocess

# %%
jp_650 = pd.read_csv('650songs.csv', index_col=None, header=None, names=['cn_singer', 'cn_song', 'jp_singer', 'jp_song'])
# only use 'cn_singer', 'cn_song' columns
jp_650 = jp_650[['jp_singer', 'jp_song']]

# %%
for id_row, content_row in jp_650.iterrows():
    print(f'Start processing {id_row+1} {content_row["jp_singer"]} {content_row["jp_song"]}')
    working_dir = Path('jp_lyrics') / str(id_row+1)
    working_dir.mkdir(parents=True, exist_ok=True)

    # use download_jp.py to save the webpages
    singer = content_row['jp_singer']
    song = content_row['jp_song']
    search_term = f'{singer} {song}'
    # subprocess.run(['python', '/home/zguo/Projects/jpzh_music/download_jp.py', search_term], cwd=working_dir)
    with open(working_dir / 'jp_lyrics.log', 'w') as log_file:
        subprocess.run(['python', '/home/zguo/Projects/jpzh_music/download_jp.py', search_term], cwd=working_dir, stdout=sys.stdout, stderr=log_file)