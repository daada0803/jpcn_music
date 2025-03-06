# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# %%
df = pd.read_csv('650songs_lyrics_cleaned.csv')
# Count the number of songs per Chinese singer
cn_singer_counts = df['cn_singer'].value_counts()

# Calculate the percentage of each singer
total_songs = cn_singer_counts.sum()
singer_percentages = (cn_singer_counts / total_songs) * 100

# Define threshold for "Others" category (singers under 1% are grouped)
threshold_others = 1  
above_threshold = singer_percentages[singer_percentages >= threshold_others]
below_threshold = singer_percentages[singer_percentages < threshold_others]

# Sum the percentages of singers below the threshold to create "Others" category
others_percentage = below_threshold.sum()

# Append "Others" category if necessary
if others_percentage > 0:
    above_threshold["Others"] = others_percentage

# Create labels only for singers above 2%, otherwise leave blank
labels = [singer if percentage > 2 else "" for singer, percentage in above_threshold.items()]

# %%
# Set Chinese font (SimHei or Microsoft YaHei)
plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']  # For Windows: Try 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are shown properly

# Plot the pie chart
# textprops = {'fontsize': 14}  # Adjust the fontsize as needed
plt.figure(figsize=(8, 8))
# plt.pie(above_threshold, labels=labels, autopct=lambda p: f'{p:.1f}%' if p > 2 else '', startangle=140)
# plt.pie(above_threshold, autopct=lambda p: f'{p:.1f}%' if p > 2 else '', startangle=140)
plt.pie(above_threshold, startangle=140)
plt.title('Distribution of Chinese Singers (Singers <1% Grouped as "Others", Only Singers >2% Labeled)')
plt.show()

# %%
jp_singer_counts = df['jp_singer'].value_counts()

# Calculate the percentage of each singer
total_songs = jp_singer_counts.sum()
singer_percentages = (jp_singer_counts / total_songs) * 100

# Define threshold for "Others" category (singers under 1% are grouped)
threshold_others = 1  
above_threshold = singer_percentages[singer_percentages >= threshold_others]
below_threshold = singer_percentages[singer_percentages < threshold_others]

# Sum the percentages of singers below the threshold to create "Others" category
others_percentage = below_threshold.sum()

# Append "Others" category if necessary
if others_percentage > 0:
    above_threshold["Others"] = others_percentage

# Create labels only for singers above 2%, otherwise leave blank
labels = [singer if percentage > 2 else "" for singer, percentage in above_threshold.items()]

# %%
# Set Chinese font (SimHei or Microsoft YaHei)
plt.rcParams['font.sans-serif'] = ['Source Han Sans']  # For Windows: Try 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are shown properly

# Plot the pie chart
plt.figure(figsize=(8, 8))
# plt.pie(above_threshold, labels=labels, autopct=lambda p: f'{p:.1f}%' if p > 2 else '', startangle=140)
# plt.pie(above_threshold, autopct=lambda p: f'{p:.1f}%' if p > 2 else '', startangle=140)
plt.pie(above_threshold, startangle=140)
plt.title('Distribution of Japanese Singers (Singers <1% Grouped as "Others", Only Singers >2% Labeled)')
plt.show()

# %%
