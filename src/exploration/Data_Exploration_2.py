#%% md
# Download and extract files
# https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set
#%%
import zipfile

zip_path = r"../../data/raw/dataset_2/archive.zip"

extract_path = r"../../data/raw/dataset_2"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

#%%
# Import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
import random
#%%
# Load in dataset
train_dir = r"../../data/raw/dataset_2/train"
data_df = pd.read_csv(r"../../data/raw/dataset_2/dogs.csv")

print(data_df.shape)

data_df.head()

#%%
# Split into training, validation, and test data
train_df = data_df[data_df.iloc[:, 2] == "train"].copy()
valid_df = data_df[data_df.iloc[:, 2] == "valid"].copy()
test_df = data_df[data_df.iloc[:, 2] == "test"].copy()
train_df.head()

#convert to csv
#train
output_folder = r"..\..\data\raw\dataset_2"
train_path = os.path.join(output_folder, 'train.csv')
train_df.to_csv(train_path, index=False)

#validation
output_folder = r"..\..\data\raw\dataset_2"
valid_path = os.path.join(output_folder, 'valid.csv')
valid_df.to_csv(valid_path, index=False)

#test
output_folder = r"..\..\data\raw\dataset_2"
test_path = os.path.join(output_folder, 'test.csv')
test_df.to_csv(test_path, index=False)
#%%
# Check the basic information about the data
print("train_df information:")
train_df.info()
print("\nvalid_df information:")
valid_df.info()
print("\ntest_df information:")
test_df.info()

#%%
# View a few rows of the data
print(train_df.head())
print(valid_df.head())
print(test_df.head())
#%% md
# # Check for missing data
#%%
# Check for missing data in each dataset
print("\nTRAIN_DF:")
print(train_df.isnull().sum())
print("\nVALID_DF:")
print(valid_df.isnull().sum())
print("\nTEST_DF:")
print(test_df.isnull().sum())
#%%
# Check the distribution of dog breeds in the training set
breed_counts = train_df['labels'].value_counts()
breed_counts.plot(kind='bar', figsize=(12, 6))
plt.title('Distribution of Dog Breeds in the Training Set')
plt.xlabel('Dog Breed')
plt.ylabel('Count')

plt.savefig(r"..\..\output\exploration\dataset_2\Distribution_Dog_Breeds_in_Training")
plt.show()
#%%

# Display a few sample images from the training set
sample_images = random.sample(list(train_df['filepaths']), 5)

for img_path in sample_images:
    img = cv2.imread(os.path.join(r"../../data/raw/dataset_2", img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(img_path)
    plt.axis('off')
    plt.show()

#%%
random.sample(list(train_df['filepaths']), 5)
#%% md
# # Calculate the Number of Images in Each Split
#%%
# Calculate the number of images in each split
image_counts = data_df['data set'].value_counts()

# Create a new DataFrame to make plotting easier
counts_df = pd.DataFrame({'split': image_counts.index, 'count': image_counts.values})

# Display the new DataFrame
print(counts_df)

#%% md
# # Plot Bar_plot and Histogram
#%%
import seaborn as sns
# Bar plot for the count of images in each dataset split
plt.figure(figsize=(10, 6))
sns.barplot(x='split', y='count', data=counts_df)
plt.title('Bar Plot of Image Counts by Dataset Split')
plt.xlabel('Dataset Split')
plt.ylabel('Image Count')

plt.savefig(r"..\..\output\exploration\dataset_2\Barplot_image_count.png")
plt.show()

#%%
# Histogram of image counts by dataset split
plt.figure(figsize=(10, 6))
sns.histplot(data=data_df, x='data set', kde=False, bins=3)
plt.title('Histogram of Image Counts by Dataset Split')
plt.xlabel('Dataset Split')
plt.ylabel('Image Count')


plt.savefig(r"..\..\output\exploration\dataset_2\Histogram_image_count.png")
plt.show()

#%%
# Calculate the number of images per category in each split
category_counts = data_df.groupby(['data set', 'labels']).size().reset_index(name='count')

# Display the new DataFrame
print(category_counts.head())

# Plot the distribution of image counts per category within each split
plt.figure(figsize=(14, 8))
sns.barplot(x='labels', y='count', hue='data set', data=category_counts)
plt.title('Bar Plot of Image Counts per Category within Each Dataset Split')
plt.xlabel('Category')
plt.ylabel('Image Count')
plt.xticks(rotation=90)

plt.savefig(r"..\..\output\exploration\dataset_2\Bar_plot_Image_count_per_Category_Each_Dataset.png")
plt.show()

#%%
category_counts
#%% md
# # Heat_map
#%%
# Pivot the data to create a matrix for the heatmap
heatmap_data = category_counts.pivot(index='labels', columns='data set', values='count').fillna(0)

# Display the new DataFrame
print(heatmap_data.head())
#%%
# Plot the heatmap

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm")
plt.title('Heatmap of Image Counts per Category within Each Dataset Split')
plt.xlabel('Dataset Split')
plt.ylabel('Category')



plt.savefig(r"..\..\output\exploration\dataset_2\Heatmap_Image_Count_Per_Category_Each_Dataset_Split.png")
plt.show()

#%%
