# %%
import pandas as pd
import numpy as np
import random
import re
import json
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import linear_kernel
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import Normalizer
# from sklearn.pipeline import make_pipeline
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import seaborn as sns


# %%
# load 650songs_lyrics.csv, keep original index in a new column 'original_index', and drop rows with NaN values
df = pd.read_csv('650songs_lyrics.csv')
df['original_index'] = df.index
df = df.dropna()

# %%
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
df_lyrics = pd.DataFrame(row_list)
df_lyrics.to_csv('650songs_lyrics_cleaned.csv', index=False)
# %%
model = SentenceTransformer('sentence-transformers/LaBSE')
# %%
embeddings_jp = model.encode(jp_lyrics_list, convert_to_tensor=True)
embeddings_cn = model.encode(cn_lyrics_list, convert_to_tensor=True)
# save embeddings to file
torch.save(embeddings_jp, 'embeddings_jp.pt')
torch.save(embeddings_cn, 'embeddings_cn.pt')
# %%
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
similarity_matrix = cos(embeddings_jp, embeddings_cn)
similarity_matrix = similarity_matrix.cpu().numpy()
# %%
plt.figure(figsize=(10, 6))
# plt.hist(similarity_matrix, bins=20, color='skyblue', edgecolor='black', range=(0, 1))
sns.histplot(similarity_matrix, bins=20, kde=True, color='skyblue', edgecolor='black')
plt.title('Cosine Similarity Distribution Between Japanese and Chinese Song Versions')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.xlim(0, 1)
plt.grid(True)
plt.show()
# %%
# Calculate actual cosine similarities
actual_cosine_similarities = cos(embeddings_jp, embeddings_cn).cpu().numpy()

# Permutation test parameters
num_permutations = 10000
permuted_similarities = []

# Perform permutations
for _ in range(num_permutations):
    # Shuffle the Chinese embeddings
    shuffled_indices = random.sample(range(len(embeddings_cn)), len(embeddings_cn))
    shuffled_cn_embeddings = embeddings_cn[shuffled_indices]
    
    # Calculate cosine similarities with shuffled embeddings
    permuted_similarity = cos(embeddings_jp, shuffled_cn_embeddings).cpu().numpy()
    permuted_similarities.extend(permuted_similarity)

# Convert to numpy array for analysis
permuted_similarities = np.array(permuted_similarities)

# %%
# Plot the histogram of permuted similarities
plt.figure(figsize=(10, 6))
plt.hist(permuted_similarities, bins=20, color='lightgray', edgecolor='black', alpha=0.7, label='Permuted Similarities', range=(0, 1))
plt.axvline(x=np.mean(actual_cosine_similarities), color='red', linestyle='--', linewidth=2, label='Actual Mean Similarity')
plt.title('Permutation Test: Cosine Similarity Distribution')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.xlim(0, 1)
plt.legend()
plt.grid(True)
plt.show()

# %%
# Calculate p-value
p_value = np.mean(permuted_similarities >= np.mean(actual_cosine_similarities))
print(f"P-value: {p_value}")
# %%
# plot the actual cosine similarities and the permuted similarities on the same plot
plt.figure(figsize=(10, 6))
sns.histplot(actual_cosine_similarities, bins=20, stat='density', kde=True, color='skyblue', alpha=0.6, label='Actual Similarities')
# Plot permuted similarities KDE
sns.kdeplot(permuted_similarities, color='red', linewidth=1, label='Permuted Similarities KDE')
plt.title('Actual Cosine Similarities and Permuted Similarities KDE')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.xlim(0, 1)  # Set x-axis limits from 0 to 1
plt.legend()
plt.grid(True)
plt.show()
# %%
embeddings_jp = embeddings_jp.cpu().numpy()
embeddings_cn = embeddings_cn.cpu().numpy()
embeddings = np.concatenate((embeddings_jp, embeddings_cn), axis=0)
# %%
pca = PCA(n_components=2)
principal_components = pca.fit_transform(embeddings)

# Create labels for the embeddings
labels = ['jp'] * len(embeddings_jp) + ['cn'] * len(embeddings_cn)

# Plot the PCA results
plt.figure(figsize=(10, 6))
for label in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(principal_components[indices, 0], principal_components[indices, 1], label=label)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Japanese and Chinese Song Lyrics Embeddings')
plt.legend()
plt.grid(True)
plt.show()
# %%
from openai import OpenAI
client = OpenAI(api_key="sk-")

# %%
jp_response = client.embeddings.create(
    input=jp_lyrics_list,
    model="text-embedding-3-large"
)
embeddings_jp_openai = [item.embedding for item in jp_response.data]
# %%
cn_response = client.embeddings.create(
    input=cn_lyrics_list,
    model="text-embedding-3-large"
)
embeddings_cn_openai = [item.embedding for item in cn_response.data]

# %%
embeddings = np.concatenate((embeddings_jp_openai, embeddings_cn_openai), axis=0)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(embeddings)

# Create labels for the embeddings
labels = ['jp'] * len(embeddings_jp) + ['cn'] * len(embeddings_cn)

# Plot the PCA results
plt.figure(figsize=(10, 6))
for label in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(principal_components[indices, 0], principal_components[indices, 1], label=label)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Japanese and Chinese Song Lyrics Embeddings using OpenAI')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Calculate actual cosine similarities
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
embeddings_jp_openai = torch.tensor(embeddings_jp_openai)
embeddings_cn_openai = torch.tensor(embeddings_cn_openai)
actual_cosine_similarities_openai = cos(embeddings_jp_openai, embeddings_cn_openai).cpu().numpy()

# Permutation test parameters
num_permutations = 10000
permuted_similarities_openai = []

# Perform permutations
for _ in range(num_permutations):
    # Shuffle the Chinese embeddings
    shuffled_indices = random.sample(range(len(embeddings_cn_openai)), len(embeddings_cn_openai))
    shuffled_cn_embeddings = embeddings_cn_openai[shuffled_indices]
    
    # Calculate cosine similarities with shuffled embeddings
    permuted_similarity = cos(embeddings_jp_openai, shuffled_cn_embeddings).cpu().numpy()
    permuted_similarities_openai.extend(permuted_similarity)
    if _ % 1000 == 0:
        print(f'Permutation {_} done.')

# Convert to numpy array for analysis
permuted_similarities_openai = np.array(permuted_similarities_openai)

# %%
# Plot
plt.figure(figsize=(10, 6))
sns.histplot(actual_cosine_similarities_openai, bins=20, stat='density', kde=True, color='skyblue', alpha=0.6, label='Actual Similarities')
# Plot permuted similarities KDE
sns.kdeplot(permuted_similarities_openai, color='red', linewidth=1, label='Permuted Similarities KDE')
plt.title('Actual Cosine Similarities and Permuted Similarities KDE using OpenAI')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.xlim(0, 1)  # Set x-axis limits from 0 to 1
plt.legend()
plt.grid(True)
plt.show()
# %%
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained multilingual BERT model and tokenizer
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)  # Assuming 6 emotion classes

# Function to predict emotion
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Example usage
jp_lyrics = "あなたのことが好きです"  # Japanese lyrics
cn_lyrics = "我喜欢你"  # Chinese lyrics

jp_emotion = predict_emotion(jp_lyrics)
cn_emotion = predict_emotion(cn_lyrics)

print(f"Predicted emotion for Japanese lyrics: {jp_emotion}")
print(f"Predicted emotion for Chinese lyrics: {cn_emotion}")

# %%
