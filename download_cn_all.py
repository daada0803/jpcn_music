# %%
import sys
import pandas as pd
from pathlib import Path
import subprocess

# %%
cn_650 = pd.read_csv('650songs.csv', index_col=None, header=None, names=['cn_singer', 'cn_song', 'jp_singer', 'jp_song'])
# only use 'cn_singer', 'cn_song' columns
cn_650 = cn_650[['cn_singer', 'cn_song']]

# %%
for id_row, content_row in cn_650.iterrows():
    print(f'Start processing {id_row+1} {content_row["cn_singer"]} {content_row["cn_song"]}')
    working_dir = Path('cn_music') / str(id_row+1)
    working_dir.mkdir(parents=True, exist_ok=True)

    # use download_jp.py to save the webpages
    singer = content_row['cn_singer']
    song = content_row['cn_song']
    search_term = f'{singer} {song}'
    # subprocess.run(['python', '/home/zguo/Projects/jpzh_music/download_jp.py', search_term], cwd=working_dir)
    with open(working_dir / 'cn_music.log', 'w') as log_file:
        subprocess.run(['python', '/home/zguo/Projects/jpzh_music/download_cn.py', search_term], cwd=working_dir, stdout=sys.stdout, stderr=log_file)
    # if id_row >= 10:
    #     break