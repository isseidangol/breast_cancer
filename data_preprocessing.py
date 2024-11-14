import os
import gc
import uuid
import shutil
import random
import glob as gb
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.special import gamma
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Initialize global constants
IMAGE_DIR = 'mri/jpeg'
DATA_CSV_DIR = 'mri/csv'
OUTPUT_IMAGE_DIR = 'mri/dataset/images/'
OUTPUT_MASK_DIR = 'mri/dataset/masks/'
OUTPUT_CSV_DIR = 'mri/dataset/annotations/'

def check_versions():
    """Print the version of each module used."""
    modules = {
        "os": os, "gc": gc, "uuid": uuid, "shutil": shutil, "random": random, "glob": gb,
        "numpy": np, "pandas": pd, "tensorflow": tf, "matplotlib": plt, "cv2": cv2,
        "tqdm": tqdm, "PIL": Image, "scipy": gamma, "keras": Sequential, "sklearn": train_test_split
    }
    for name, mod in modules.items():
        print(f"{name}:", mod.__version__ if hasattr(mod, "__version__") else "No version info")

def load_data():
    """Load and filter data based on Series Description."""
    di_data = pd.read_csv(os.path.join(DATA_CSV_DIR, 'dicom_info.csv'))
    mammo_data = di_data[di_data['SeriesDescription'] == 'full mammogram images']
    roi_data = di_data[di_data['SeriesDescription'] == 'ROI mask images']
    cropped_data = di_data[di_data['SeriesDescription'] == 'cropped images']
    
    # Replace paths
    for data, name in zip([mammo_data, roi_data, cropped_data], ['mammo', 'roi', 'cropped']):
        data['image_path'] = data['image_path'].str.replace('CBIS-DDSM/jpeg', IMAGE_DIR)
        data = data[['image_path', 'PatientID', 'SeriesDescription']]
        print(f'{name} image paths:\n{data.head(10)}')
    
    return mammo_data, roi_data, cropped_data

def display_images(data, title="Image"):
    """Display images with grayscale conversion."""
    for file in data['image_path'][:10]:
        try:
            img = Image.open(file).convert("L")
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.show()
        except Exception as e:
            print(f"Error opening image: {e}")

def prepare_directories():
    """Create output directories for images and masks."""
    for label in ['malignant', 'benign']:
        os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, label), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_MASK_DIR, label), exist_ok=True)
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

def process_rois(calc_train, roi_filtered_data):
    """Process ROI images and save mask files."""
    rois = []
    for _, row in tqdm(calc_train.iterrows(), total=calc_train.shape[0]):
        pathology = row['pathology']
        mask_file_path = row['image_path']
        
        if pathology == "MALIGNANT":
            label, class_label = 'malignant', 1
        elif pathology in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
            label, class_label = 'benign', 0
        else:
            continue
        
        mask_output_path = os.path.join(OUTPUT_MASK_DIR, label, f"{row['mammo_name']}_mask.jpg")
        if os.path.exists(mask_file_path):
            shutil.copy(mask_file_path, mask_output_path)
            mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    rois.append([f"{row['mammo_name']}_image.jpg", class_label, x, y, x + w, y + h])
    
    df_rois = pd.DataFrame(rois, columns=['image', 'label', 'x_min', 'y_min', 'x_max', 'y_max'])
    df_rois.to_csv(os.path.join(OUTPUT_CSV_DIR, "roi_annotations.csv"), index=False)
    print("ROI annotations saved.")

def process_images(calc_train, mammo_data):
    """Process mammography images and copy to output directories."""
    for _, row in tqdm(calc_train.iterrows(), total=calc_train.shape[0]):
        pathology = row['pathology']
        if pathology == "MALIGNANT":
            label = 'malignant'
        elif pathology in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
            label = 'benign'
        else:
            continue
        
        image_output_path = os.path.join(OUTPUT_IMAGE_DIR, label, f"{row['PatientID']}_image.jpg")
        if os.path.exists(row['image_path']):
            shutil.copy(row['image_path'], image_output_path)
        else:
            print(f"Image file not found: {row['image_path']}")

def main():
    check_versions()
    prepare_directories()
    mammo_data, roi_data, cropped_data = load_data()
    display_images(mammo_data, "Mammogram Image")
    display_images(roi_data, "ROI Image")
    display_images(cropped_data, "Cropped Image")
    
    # Load and preprocess datasets
    calc_train = pd.read_csv(os.path.join(DATA_CSV_DIR, 'calc_case_description_train_set.csv'))
    calc_train['mammo_name'] = calc_train['image file path'].str.split('/').str[0]
    calc_train['roi_name'] = calc_train['ROI mask file path'].str.split('/').str[0]
    calc_train['cropped_name'] = calc_train['cropped image file path'].str.split('/').str[0]
    
    # Merge datasets for processing
    mammo_filtered_data = pd.merge(calc_train, mammo_data, left_on="mammo_name", right_on="PatientID", how="inner")
    roi_filtered_data = pd.merge(calc_train, roi_data, left_on="roi_name", right_on="PatientID", how="inner")
    
    process_rois(calc_train, roi_filtered_data)
    process_images(calc_train, mammo_filtered_data)
    
if __name__ == "__main__":
    main()
import os
import gc
import uuid
import shutil
import random
import glob as gb
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.special import gamma
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Initialize global constants
IMAGE_DIR = 'mri/jpeg'
DATA_CSV_DIR = 'mri/csv'
OUTPUT_IMAGE_DIR = 'mri/dataset/images/'
OUTPUT_MASK_DIR = 'mri/dataset/masks/'
OUTPUT_CSV_DIR = 'mri/dataset/annotations/'

def check_versions():
    """Print the version of each module used."""
    modules = {
        "os": os, "gc": gc, "uuid": uuid, "shutil": shutil, "random": random, "glob": gb,
        "numpy": np, "pandas": pd, "tensorflow": tf, "matplotlib": plt, "cv2": cv2,
        "tqdm": tqdm, "PIL": Image, "scipy": gamma, "keras": Sequential, "sklearn": train_test_split
    }
    for name, mod in modules.items():
        print(f"{name}:", mod.__version__ if hasattr(mod, "__version__") else "No version info")

def load_data():
    """Load and filter data based on Series Description."""
    di_data = pd.read_csv(os.path.join(DATA_CSV_DIR, 'dicom_info.csv'))
    mammo_data = di_data[di_data['SeriesDescription'] == 'full mammogram images']
    roi_data = di_data[di_data['SeriesDescription'] == 'ROI mask images']
    cropped_data = di_data[di_data['SeriesDescription'] == 'cropped images']
    
    # Replace paths
    for data, name in zip([mammo_data, roi_data, cropped_data], ['mammo', 'roi', 'cropped']):
        data['image_path'] = data['image_path'].str.replace('CBIS-DDSM/jpeg', IMAGE_DIR)
        data = data[['image_path', 'PatientID', 'SeriesDescription']]
        print(f'{name} image paths:\n{data.head(10)}')
    
    return mammo_data, roi_data, cropped_data

def display_images(data, title="Image"):
    """Display images with grayscale conversion."""
    for file in data['image_path'][:10]:
        try:
            img = Image.open(file).convert("L")
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.show()
        except Exception as e:
            print(f"Error opening image: {e}")

def prepare_directories():
    """Create output directories for images and masks."""
    for label in ['malignant', 'benign']:
        os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, label), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_MASK_DIR, label), exist_ok=True)
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

def process_rois(calc_train, roi_filtered_data):
    """Process ROI images and save mask files."""
    rois = []
    for _, row in tqdm(calc_train.iterrows(), total=calc_train.shape[0]):
        pathology = row['pathology']
        mask_file_path = row['image_path']
        
        if pathology == "MALIGNANT":
            label, class_label = 'malignant', 1
        elif pathology in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
            label, class_label = 'benign', 0
        else:
            continue
        
        mask_output_path = os.path.join(OUTPUT_MASK_DIR, label, f"{row['mammo_name']}_mask.jpg")
        if os.path.exists(mask_file_path):
            shutil.copy(mask_file_path, mask_output_path)
            mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    rois.append([f"{row['mammo_name']}_image.jpg", class_label, x, y, x + w, y + h])
    
    df_rois = pd.DataFrame(rois, columns=['image', 'label', 'x_min', 'y_min', 'x_max', 'y_max'])
    df_rois.to_csv(os.path.join(OUTPUT_CSV_DIR, "roi_annotations.csv"), index=False)
    print("ROI annotations saved.")

def process_images(calc_train, mammo_data):
    """Process mammography images and copy to output directories."""
    for _, row in tqdm(calc_train.iterrows(), total=calc_train.shape[0]):
        pathology = row['pathology']
        if pathology == "MALIGNANT":
            label = 'malignant'
        elif pathology in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
            label = 'benign'
        else:
            continue
        
        image_output_path = os.path.join(OUTPUT_IMAGE_DIR, label, f"{row['PatientID']}_image.jpg")
        if os.path.exists(row['image_path']):
            shutil.copy(row['image_path'], image_output_path)
        else:
            print(f"Image file not found: {row['image_path']}")

def main():
    check_versions()
    prepare_directories()
    mammo_data, roi_data, cropped_data = load_data()
    display_images(mammo_data, "Mammogram Image")
    display_images(roi_data, "ROI Image")
    display_images(cropped_data, "Cropped Image")
    
    # Load and preprocess datasets
    calc_train = pd.read_csv(os.path.join(DATA_CSV_DIR, 'calc_case_description_train_set.csv'))
    calc_train['mammo_name'] = calc_train['image file path'].str.split('/').str[0]
    calc_train['roi_name'] = calc_train['ROI mask file path'].str.split('/').str[0]
    calc_train['cropped_name'] = calc_train['cropped image file path'].str.split('/').str[0]
    
    # Merge datasets for processing
    mammo_filtered_data = pd.merge(calc_train, mammo_data, left_on="mammo_name", right_on="PatientID", how="inner")
    roi_filtered_data = pd.merge(calc_train, roi_data, left_on="roi_name", right_on="PatientID", how="inner")
    
    process_rois(calc_train, roi_filtered_data)
    process_images(calc_train, mammo_filtered_data)
    
if __name__ == "__main__":
    main()
