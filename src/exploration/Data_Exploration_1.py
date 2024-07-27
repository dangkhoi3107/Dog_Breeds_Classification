#%% md
# 
# Download and extract files
# https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
#%%
import zipfile

zip_path = r"..\..\data\raw\Dog_standford.zip"  

extract_path = r"..\..\data\raw"  

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

#%% md
# # Data exploration 
# 
#%%
import os

#list_breed_folders: contain a list of breeds
#for each breed: contain a list of images
#-----------------------------------------
#list_annotation_folders: contain a list of breeds
#for each breed: contain a list of annotations

list_image_folder_paths = os.path.join(extract_path, 'images\Images')


list_breed_folders = os.listdir(list_image_folder_paths)


list_annotation_folder_paths = os.path.join(extract_path, 'annotations\Annotation')

list_annotation_folders = os.listdir(list_annotation_folder_paths)

print(f"List Breed Folders: {list_breed_folders}")
print("\n")
print(f"List Annotation Folder: {list_annotation_folders}")


#%%
from PIL import Image
# lists contain all the images/ annotation files 
list_images =[]
list_annotations=[]

#path for vissualize/read xml files
list_image_paths=[]
list_ann_paths=[]

# list image size
list_image_sizes=[]

# append all image
for folder in list_breed_folders:
    folder_path= os.path.join(extract_path,'images\Images',folder)
    tmp_list_img_path = os.listdir(folder_path) # This only contains 1 breed at 1 loop

    for img in tmp_list_img_path:
        img_path = os.path.join(folder_path, img)
        list_images.append(img)
        list_image_paths.append(img_path)
        
        # Get size of image
        with Image.open(img_path) as image:
            list_image_sizes.append(image.size)  # return (width, height)
        
# append all annotation files
for folder in list_annotation_folders:
    folder_path= os.path.join(extract_path,'annotations\Annotation',folder)
    tmp_list_ann_path = os.listdir(folder_path) # This only contains 1 breed at 1 loop
    
    for ann in tmp_list_ann_path:
        list_annotations.append(ann)
        
        # Store all the ann paths for long-term using
        list_ann_paths.append(os.path.join(folder_path,ann))
        
print(f"A sample in List image names: {list_images[0]}")
print(f"A sample in List annotation files: {list_annotations[0]}")
print(f"A sample in List Image Paths: {list_image_paths[0]}")
print(f"A sample in List Annotation Paths: {list_ann_paths[0]}")


#%% md
# # Check for image/ XML files
#%%
import matplotlib.pyplot as plt
from PIL import Image
# import numpy as np

# Vissualize image and print a sample folder
img = Image.open(list_image_paths[0])
plt.imshow(img)
plt.title(list_image_paths[0])
plt.show()
#%%
# Check for XML file
import xml.etree.ElementTree as ET

print("Details of an Annotation file: \n")
tree = ET.parse(list_ann_paths[0])
ET.dump(tree)


#%% md
# # Check number, size of data
#%%
import numpy as np

# Check for number of breeds
num_breeds = len(list_breed_folders)
print(f"The number of breeds is :{num_breeds} \n")

# Check for number of images
num_images = len(list_images)

dic_breed_count = {}
for breed_folder in list_breed_folders:
    folder_path = os.path.join(list_image_folder_paths, breed_folder)
    dic_breed_count[breed_folder] = len(os.listdir(folder_path))
    
# print the number of image for each breed in a table format
print(f"{'Breed':<40} {'Number of Images':<20} {'Total Number of Breeds':<25}")
print('-' * 85)
first_row = True
for breed, count in dic_breed_count.items():
    if first_row:
        print(f"{breed:<40} {count:<20} {num_breeds:<25}")
        first_row = False
    else:
        print(f"{breed:<40} {count:<20} {'':<25}")
        
        

#%%
# Calculate the average image size
img_size = np.array(list_image_sizes)
print(f'Average image size: {img_size.mean(axis=0)}')
#%% md
# # Box plot of the data
#%%
image_counts = list(dic_breed_count.values())

# box plot
plt.figure(figsize=(12, 8))
plt.boxplot(image_counts, vert=False, patch_artist=True)
plt.xlabel('Number of Images')
plt.title('Distribution of Number of Images for Each Dog Breed')
plt.show()
#%% md
# # Histogram of the data
#%%
import seaborn as sns
#  histogram
plt.figure(figsize=(12, 8))
sns.histplot(image_counts, bins=20, kde=True)
plt.xlabel('Number of Images')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Images for Each Dog Breed')
plt.show()
#%% md
# # Find missing data
#%%
# Find missing value
missing_data_images = [img for img in list_image_paths if not os.path.exists(img)]
missing_data_annotations = [ann for ann in list_ann_paths if not os.path.exists(ann)]

print(f"Missing images: {len(missing_data_images)}")
print(f"Missing annotations: {len(missing_data_annotations)}")

# Handle missing value 
list_image_paths = [img for img in list_image_paths if os.path.exists(img)]
list_ann_paths = [ann for ann in list_ann_paths if os.path.exists(ann)]



#%% md
# # Check duplicate data
#%%
unique_image_paths = set(list_image_paths)
unique_ann_paths = set(list_ann_paths)

print(f"Unique images: {len(unique_image_paths)}")
print(f"Unique annotations: {len(unique_ann_paths)}")

# Handle duplicate data
list_image_paths = list(unique_image_paths)
list_ann_paths = list(unique_ann_paths)
#%% md
# # Distribute data by label
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Sử dụng dic_breed_count để vẽ biểu đồ phân phối nhãn
plt.figure(figsize=(20,10))
sns.barplot(x=list(dic_breed_count.keys()), y=list(dic_breed_count.values()))
plt.xticks(rotation=90)
plt.title('Distribution of Dog Breeds in Dataset')
plt.xlabel('Dog Breed')
plt.ylabel('Count')
plt.show()

#%% md
# # Annotation Data Analysis
#%%
import matplotlib.patches as patches

image_folder = os.path.join(extract_path, 'images/Images')
annotation_folder = os.path.join(extract_path, 'annotations/Annotation')

list_breed_folders = os.listdir(image_folder)

# Iterate through each breed folder
for breed_folder in list_breed_folders:
    image_breed_folder = os.path.join(image_folder, breed_folder)
    annotation_breed_folder = os.path.join(annotation_folder, breed_folder)
    
    # Get the list of image and annotation files
    image_files = os.listdir(image_breed_folder)
    annotation_files = os.listdir(annotation_breed_folder)
    
    # Sort the lists to ensure correct matching
    image_files.sort()
    annotation_files.sort()
    
    # Check the number of files
    print(f"Number of images in {breed_folder}: {len(image_files)}")
    print(f"Number of annotations in {breed_folder}: {len(annotation_files)}")
    
    # Display the first 5 images and their bounding boxes
    for img_file, ann_file in zip(image_files[:5], annotation_files[:5]):
        img_path = os.path.join(image_breed_folder, img_file)
        ann_path = os.path.join(annotation_breed_folder, ann_file)
        
        img = Image.open(img_path)
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        bbox = root.find('object').find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        plt.title(img_file)
        plt.show()

    # Stop after displaying the first 5 images of the first folder
    break
#%% md
# # Check the size of the bounding boxes
#%%
import pandas as pd

# calculating bounding box sizes
bbox_sizes = []
for ann_path in list_ann_paths:
    tree = ET.parse(ann_path)
    root = tree.getroot()
    
    bbox = root.find('object').find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    width = xmax - xmin
    height = ymax - ymin
    bbox_sizes.append((width, height))

# Create a DataFrame for better analysis
bbox_df = pd.DataFrame(bbox_sizes, columns=['Width', 'Height'])

# Display basic statistics
print(bbox_df.describe())

# Plot the distribution of bounding box sizes
plt.figure(figsize=(12, 6))
sns.histplot(bbox_df['Width'], kde=True, bins=30, color='blue', label='Width')
sns.histplot(bbox_df['Height'], kde=True, bins=30, color='green', label='Height')
plt.xlabel('Bounding Box Size')
plt.ylabel('Frequency')
plt.title('Distribution of Bounding Box Sizes')
plt.legend()
plt.show()

#%% md
# # Correlation Analysis (Widht/ Height)
#%%

correlation = bbox_df.corr()
print("Correlation matrix:\n", correlation)

sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Bounding Box Dimensions')
plt.show()

#%%
import numpy as np

# Check for number of breeds
num_breeds = len(list_breed_folders)
print(f"The number of breeds is :{num_breeds} \n")

# Check for number of images
num_images = len(list_images)

dic_breed_count = {}
for breed_folder in list_breed_folders:
    folder_path = os.path.join(list_image_folder_paths, breed_folder)
    dic_breed_count[breed_folder] = len(os.listdir(folder_path))
    
# print the number of image for each breed in a table format
print(f"{'Breed':<40} {'Number of Images':<20} {'Total Number of Breeds':<25}")
print('-' * 85)
first_row = True
for breed, count in dic_breed_count.items():
    if first_row:
        print(f"{breed:<40} {count:<20} {num_breeds:<25}")
        first_row = False
    else:
        print(f"{breed:<40} {count:<20} {'':<25}")
        
# Calculate the average image size
img_size = np.array(list_image_sizes)
print(f'Average image size: {img_size.mean(axis=0)}')

image_counts = list(dic_breed_count.values())

# box plot
plt.figure(figsize=(12, 8))
plt.boxplot(image_counts, vert=False, patch_artist=True)
plt.xlabel('Number of Images')
plt.title('Distribution of Number of Images for Each Dog Breed')
plt.show()

# histogram
plt.figure(figsize=(12, 8))
sns.histplot(image_counts, bins=20, kde=True)
plt.xlabel('Number of Images')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Images for Each Dog Breed')
plt.show()

#%%
