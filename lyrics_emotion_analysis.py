# %%
import pandas as pd
import MeCab
import jieba
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import matplotlib
# Change the default font size
matplotlib.rcParams.update({'font.size': 14})
import torch
import numpy as np
import random


# %%
def load_lexicon(file_path):
    lexicon_df = pd.read_csv(file_path, sep='\t')
    emotion_dict = {}
    for _, row in lexicon_df.iterrows():
        word = row['word']
        emotions = row[['joy', 'anger', 'sadness', 'fear', 'disgust']].to_dict()
        emotion_dict[word] = emotions
    return emotion_dict

def load_large_lexicon(file_path):
    """Load large lexicon efficiently using Pandas with multi-threading."""
    lexicon_df = pd.read_csv(file_path, sep='\t', engine='pyarrow')  # Pyarrow speeds up CSV parsing
    lexicon_df = lexicon_df.groupby("word").mean().reset_index()
    emotion_dict = lexicon_df.set_index('word').to_dict(orient='index')
    return emotion_dict

def tokenize_japanese(text):
    mecab = MeCab.Tagger()
    node = mecab.parseToNode(text)
    tokens = []
    while node:
        token = node.surface
        if token:
            tokens.append(token)
        node = node.next
    return tokens

def tokenize_chinese(text):
    tokens = jieba.lcut(text)
    return tokens

def analyze_emotions(tokens, emotion_dict):
    emotion_scores = defaultdict(float)
    for token in tokens:
        if token in emotion_dict:
            for emotion, score in emotion_dict[token].items():
                emotion_scores[emotion] += score
    return dict(emotion_scores)

# %%
# Example lyrics
# japanese_lyrics = "あなたのことが好きです"
# chinese_lyrics = "我喜欢你"

# Tokenize lyrics
# jp_tokens = tokenize_japanese(japanese_lyrics)
# cn_tokens = tokenize_chinese(chinese_lyrics)

# Load the appropriate lexicon
jp_lexicon_path = '/home/zguo/Projects/jpzh_music/mtl_grouped/ja.tsv'
cn_lexicon_path = '/home/zguo/Projects/jpzh_music/mtl_grouped/zh.tsv'
jp_emotion_dict = load_large_lexicon(jp_lexicon_path)
cn_emotion_dict = load_large_lexicon(cn_lexicon_path)

# %%
# Analyze emotions
# jp_emotion_scores = analyze_emotions(jp_tokens, jp_emotion_dict)
# cn_emotion_scores = analyze_emotions(cn_tokens, cn_emotion_dict)

# print("Japanese Lyrics Emotion Scores:", jp_emotion_scores)
# print("Chinese Lyrics Emotion Scores:", cn_emotion_scores)

# %%
df = pd.read_csv('650songs_lyrics_cleaned.csv')

jp_lyrics_list = []
cn_lyrics_list = []
row_list = []

for idx, row in df.iterrows():
    jp_lyrics_file = row['jp_lyrics']
    cn_lyrics_file = row['cn_lyrics']
    jp_lyrics = ''
    cn_lyrics = ''
    try:
        with open(jp_lyrics_file, 'r', encoding='utf-8') as f:
            jp_lyrics = json.load(f)
            jp_lyrics = jp_lyrics['lyrics']
        with open(cn_lyrics_file, 'r', encoding='utf-8') as f:
            # skip first line, which is the title of the song
            next(f)
            # read the rest of the file into a string and replace ```
            for line in f:
                line = line.strip().replace('```', '')
                line = line.replace('\'\'\'', '')
                line = line.replace('歌词', '')
                cn_lyrics += line + '\n'
            cn_lyrics = cn_lyrics.strip()
        if len(jp_lyrics) > 20 and len(cn_lyrics) > 20:
            jp_lyrics_list.append(jp_lyrics)
            cn_lyrics_list.append(cn_lyrics)
            row_list.append(row)
        else:
            raise Exception('No lyrics found')
        # print(jp_lyrics)
        # print(cn_lyrics)
    except Exception as e:
        print(f'Error loading lyrics for {idx}: {e}')
# %%
jp_emotion_list = []
cn_emotion_list = []

for jp_lyrics, cn_lyrics in zip(jp_lyrics_list, cn_lyrics_list):
    jp_tokens = tokenize_japanese(jp_lyrics)
    cn_tokens = tokenize_chinese(cn_lyrics)
    jp_emotion_scores = analyze_emotions(jp_tokens, jp_emotion_dict)
    cn_emotion_scores = analyze_emotions(cn_tokens, cn_emotion_dict)
    jp_emotion_list.append(jp_emotion_scores)
    cn_emotion_list.append(cn_emotion_scores)

# %%
df_jp_emotion = pd.DataFrame(jp_emotion_list)
df_cn_emotion = pd.DataFrame(cn_emotion_list)
df_jp_emotion = df_jp_emotion.fillna(0)
df_cn_emotion = df_cn_emotion.fillna(0)
df_jp_emotion.to_csv('650songs_jp_emotion.csv', index=False)
df_cn_emotion.to_csv('650songs_cn_emotion.csv', index=False)
# %%
# draw a scatter plot of the emotions for columns 'joy' and 'sadness'
# Add a column to distinguish the datasets
df_jp_emotion['Language'] = 'Japanese'
df_cn_emotion['Language'] = 'Chinese'

# Combine the DataFrames
df_combined = pd.concat([df_jp_emotion, df_cn_emotion])

# %%
# Set up the matplotlib figure
plt.figure(figsize=(12, 6))
# Plot density plots for 'joy'
sns.kdeplot(data=df_combined, x='joy', hue='Language', fill=True, common_norm=False, alpha=0.5)
plt.title('Density Plot of Joy Scores')
plt.xlabel('Joy Score')
plt.ylabel('Density')
plt.show()

# %%
# Plot density plots for 'sadness'
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df_combined, x='sadness', hue='Language', fill=True, common_norm=False, alpha=0.5)
plt.title('Density Plot of Sadness Scores')
plt.xlabel('Sadness Score')
plt.ylabel('Density')
plt.show()

# %%
def plot_density_plot(df, column, title, xlabel, xlim=None):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df, x=column, hue='Language', fill=True, common_norm=False, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    if xlim:
        plt.xlim(xlim)
    plt.show()

# %%
plot_density_plot(df_combined, 'anger', 'Density Plot of Anger Scores', 'Anger Score')
plot_density_plot(df_combined, 'fear', 'Density Plot of Fear Scores', 'Fear Score')
plot_density_plot(df_combined, 'disgust', 'Density Plot of Disgust Scores', 'Disgust Score')
# %%
# # Set up the matplotlib figure
# plt.figure(figsize=(12, 6))

# # Plot box plots for 'joy'
# sns.boxplot(data=df_combined, x='Language', y='joy')
# plt.title('Box Plot of Joy Scores')
# plt.xlabel('Language')
# plt.ylabel('Joy Score')
# plt.show()

# # Plot box plots for 'sadness'
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=df_combined, x='Language', y='sadness')
# plt.title('Box Plot of Sadness Scores')
# plt.xlabel('Language')
# plt.ylabel('Sadness Score')
# plt.show()

# %%
# sns.scatterplot(data=df_combined, x='joy', y='sadness', hue='Language', alpha=0.7)

# # Add titles and labels
# plt.title('Scatter Plot of Joy vs. Sadness Scores')
# plt.xlabel('Joy Score')
# plt.ylabel('Sadness Score')

# # Display the plot
# plt.show()
# %%
# df_combined_normalized = df_combined.copy()
# scaler = MinMaxScaler()
# df_combined_normalized = df_combined_normalized.drop(columns='Language')
# df_combined_normalized = pd.DataFrame(scaler.fit_transform(df_combined_normalized), columns=df_combined_normalized.columns)
# df_combined_normalized['Language'] = df_combined['Language'].values
# # df_combined_normalized.to_csv('650songs_emotion_normalized.csv', index=False)

# %%
# normalized each row of the DataFrame with square root normalization
df_combined_normalized = df_combined.copy()
df_combined_normalized = df_combined_normalized.drop(columns='Language')
# df_combined_normalized = df_combined_normalized.drop(columns=['valence', 'arousal', 'dominance', 'Language'])
df_combined_normalized = pd.DataFrame(normalize(df_combined_normalized, norm='l2', axis=1), columns=df_combined_normalized.columns)
df_combined_normalized['Language'] = df_combined['Language'].values
# df_combined_normalized.to_csv('650songs_emotion_normalized.csv', index=False)

# %%
# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Create a scatter plot
sns.scatterplot(data=df_combined_normalized, x='joy', y='sadness', hue='Language', alpha=0.7)

# Add titles and labels
plt.title('Normalized Scatter Plot of Joy vs. Sadness Scores')
plt.xlabel('Normalized Joy Score')
plt.ylabel('Normalized Sadness Score')
# plt.xlim(0.2, 0.3)
# plt.ylim(0.1, 0.2)

# Display the plot
plt.show()
# %%
plt.figure(figsize=(10, 6))

# Create a scatter plot
sns.scatterplot(data=df_combined_normalized, x='anger', y='fear', hue='Language', alpha=0.7)

# Add titles and labels
plt.title('Normalized Scatter Plot of Anger vs. Fear Scores')
plt.xlabel('Normalized Anger Score')
plt.ylabel('Normalized Fear Score')

# Display the plot
plt.show()

# %%
plot_density_plot(df_combined_normalized, 'joy', 'Density Plot of Joy Scores', 'Joy Score')
plot_density_plot(df_combined_normalized, 'sadness', 'Density Plot of Sadness Scores', 'Sadness Score')
# %%
# Extract the emotion scores for Japanese and Chinese lyrics
jp_emotion_scores = df_jp_emotion.drop(columns='Language')
cn_emotion_scores = df_cn_emotion.drop(columns='Language')
jp_emotion_scores = torch.tensor(normalize(jp_emotion_scores, norm='l2', axis=1)) # Normalize each row
cn_emotion_scores = torch.tensor(normalize(cn_emotion_scores, norm='l2', axis=1)) # Normalize each row

# Calculate the cosine similarity
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
cosine_similarities = cos(jp_emotion_scores, cn_emotion_scores).cpu().numpy()

print("Cosine Similarities:", cosine_similarities)
# %%
plt.figure(figsize=(10, 6))
# plt.hist(similarity_matrix, bins=20, color='skyblue', edgecolor='black', range=(0, 1))
sns.histplot(cosine_similarities, bins=20, kde=True, color='skyblue', edgecolor='black')
plt.title('Cosine Similarity Distribution Between Japanese and Chinese Song Versions')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
# plt.xlim(0, 1)
plt.grid(True)
plt.show()
# %%
num_permutations = 1000
permuted_similarities = []

for _ in range(num_permutations):
    # Shuffle the Chinese emotion scores
    shuffled_indices = random.sample(range(len(cn_emotion_scores)), len(cn_emotion_scores))
    shuffled_cn_emotion_scores = cn_emotion_scores[shuffled_indices]
    
    # Calculate cosine similarities with shuffled emotion scores
    permuted_similarity = cos(jp_emotion_scores, shuffled_cn_emotion_scores).cpu().numpy()
    permuted_similarities.extend(permuted_similarity)
    if _ % 100 == 0:
        print(f'Permutation {_} done')

# Convert to numpy array for analysis
permuted_similarities = np.array(permuted_similarities)

# %%
plt.figure(figsize=(10, 6))
sns.histplot(cosine_similarities, bins=20, stat='density', kde=True, color='skyblue', alpha=0.6, label='Actual Similarities')
sns.kdeplot(permuted_similarities, color='red', linewidth=1, label='Permuted Similarities KDE')
plt.title('Actual Cosine Similarities and Permuted Similarities KDE')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.xlim(0, 1)
plt.legend()
plt.grid(True)
plt.show()
# %%
permuted_similarities