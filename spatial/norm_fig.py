import os
import openslide
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import scanpy as sc
import pandas as pd


def standardize_whole_image(image_np, normalizer):
    """
    Standardizes a whole image using a given normalizer.  Assumes the normalizer
    has a 'transform' method that takes a NumPy array as input.

    Args:
        image_np:  A NumPy array representing the image (H, W, C).
        normalizer:  The normalizer object (e.g., from staintools).

    Returns:
        A PIL Image object representing the standardized image.
    """
    # Apply the normalization.  The normalizer should handle the color channel ordering.
    standardized_image_np = normalizer.transform(image_np)

    # Convert back to a PIL Image for saving.  Ensure uint8 data type.
    standardized_image = Image.fromarray(standardized_image_np.astype(np.uint8))
    return standardized_image



with open('ckp/normalizer.pkl', 'rb') as f:
     normalizer = pickle.load(f)


csv_file_path = 'ORIONCRC_dataset_tile_20x/train_dataframe.csv'
df = pd.read_csv(csv_file_path)

for index, row in df.iterrows():
    image_path = row['image_path']
    base_name, _ = os.path.splitext(os.path.basename(image_path))
    image = Image.open(image_path)

    try:
        image_np = np.array(image)
        standardized_image = standardize_whole_image(image_np, normalizer)
        new_filename = base_name + '.tif'
        standardized_image.save(os.path.join('new_he', new_filename))

    except Exception as e:
        print(f"Exception occurred for image {image_path}: {e}")
        new_filename = base_name + '.tif'
        image.save(os.path.join('new_he', new_filename))

columns = ['new_filename', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3', 'CD8a', 'CD45RO', 'CD20', 
           'PD-L1', 'CD3e', 'CD163', 'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA']
new_data = []

for index, row in df.iterrows():
    target_path = row['target_path']
    nuclei_path = row['nuclei_path']
    
    target_image = Image.open(target_path)
    target_np = np.array(target_image)

    nuclei_image = Image.open(nuclei_path)
    nuclei_np = np.array(nuclei_image)

    mask = nuclei_np > 0
    means = target_np[mask].mean(axis=0)[1:17]
    new_row = [base_name + '.tif'] + means.tolist()
    new_data.append(new_row)

new_df = pd.DataFrame(new_data, columns=columns)
new_csv_path = 'ORIONCRC_dataset_tile_20x/train_dataframe_expression.csv'
new_df.to_csv(new_csv_path, index=False)