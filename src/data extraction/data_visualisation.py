import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# File directories
raw_csv = ROOT / "data/extracted_features/hand_landmarks.csv"
clean_csv = ROOT / "data/extracted_features/hand_landmarks_sanitised.csv"

# load the datasets
raw_df =  pd.read_csv(raw_csv)
cleaned_df = pd.read_csv(clean_csv)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#-------------------------------------------------
# plot raw data

# We use x0 and y0 which represent the wrist coordinates in the image
sns.scatterplot(
    data=raw_df, 
    x='x0', 
    y='y0', 
    hue='label', 
    ax=axes[0], 
    palette='viridis', 
    alpha=0.6, 
    legend='brief'
)
axes[0].set_title('Raw Data: Wrist Position (x0, y0)\n(Hand locations in image space)')
axes[0].invert_yaxis() # Invert Y because image coordinates start from top-left
axes[0].set_xlabel('Wrist X (Normalized Image Coord)')
axes[0].set_ylabel('Wrist Y (Normalized Image Coord)')

#-------------------------------------------------
# plot cleaned data

# We use f3 and f4 (Landmark 1 - Thumb CMC) to show the hand shape relative to the wrist.
sns.scatterplot(
    data=cleaned_df, 
    x='f3', 
    y='f4', 
    hue='label', 
    ax=axes[1], 
    palette='viridis', 
    alpha=0.6, 
    legend=False # Legend is redundant here
)
axes[1].set_title('Sanitised Data: Landmark 1 Relative to Wrist (f3, f4)\n(Normalized Feature Space)')
axes[1].invert_yaxis()
axes[1].set_xlabel('Relative X (Scaled)')
axes[1].set_ylabel('Relative Y (Scaled)')
#-------------------------------------------------
# Adjust layout and save
plt.tight_layout()
plt.savefig('data/scatter_comparison.png')
plt.show()

# Calculate rows cleaned
rows_removed = len(raw_df) - len(cleaned_df)
print(f"Original Rows: {len(raw_df)}")
print(f"Sanitised Rows: {len(cleaned_df)}")
print(f"Total Rows Removed: {rows_removed}")

#-------------------------------------------------
# Calcualte the class distributions

# Calculate the counts for both datasets
raw_counts = raw_df['label'].value_counts().sort_index()
clean_counts = cleaned_df['label'].value_counts().sort_index()

# Transform data into long format
dist_df = pd.DataFrame({
    'Label': raw_counts.index,
    'Raw': raw_counts.values,
    'Sanitised': clean_counts.values
}).melt(id_vars='Label', var_name='Dataset', value_name='Count')

# Create bar graph
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

sns.barplot(
    data=dist_df, 
    x='Label', 
    y='Count', 
    hue='Dataset', 
    palette=['#482677FF', '#20A387FF'] # Matching the Viridis-style palette
)

# Labels and visualisation formatting
plt.title('Class Distribution: Raw vs Sanitised Data', fontsize=15)
plt.xlabel('ASL Sign Label', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=0) # Keeps labels horizontal (A, B, C...)
plt.legend(title='Dataset')

# Label totals ontop of bar
for container in plt.gca().containers:
    plt.gca().bar_label(container, padding=3, fontsize=9)

plt.tight_layout()
plt.savefig('data/class_distribution_comparison.png')
plt.show()

# Print summary to console
print("Summary of Data Loss per Class:")
summary = pd.DataFrame({
    'Raw': raw_counts,
    'Sanitised': clean_counts,
    'Lost': raw_counts - clean_counts
})
print(summary)