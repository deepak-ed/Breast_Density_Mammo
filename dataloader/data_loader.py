import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MammographyDataset(Dataset):

    def __init__(self, split_file, data_folder, view_position, task, image_height, image_width, transform=None):

        self.mammography = pd.read_csv(split_file)
        self.data_folder = data_folder
        self.task = task
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        if view_position == 'CC':
            self.mammography = self.mammography[self.mammography['view_position'] == 'CC']
            self.mammography= self.mammography.reset_index(drop=True)
        elif view_position == 'MLO':
            self.mammography = self.mammography[self.mammography['view_position'] == 'MLO']
            self.mammography= self.mammography.reset_index(drop=True)
        elif view_position == 'both':
            self.mammography = self.mammography

    def __len__(self):
        return len(self.mammography)

    def class_name_to_labels(self):
        if self.task == 'BreastDensity':
            labels = self.mammography['breast_density'].replace({'DENSITY A': 0, 'DENSITY B': 0, 'DENSITY C': 1, 'DENSITY D': 1})
            #class_name = self.mammography.iloc[idx, 8]
            #if class_name in ['DENSITY A', 'DENSITY B']:
            #    labels = 1.0
            #elif class_name in ['DENSITY C', 'DENSITY D']:
            #    labels = 0.0
            return labels

        elif self.task == 'BIRADS':
            labels = self.mammography['breast_birads'].replace({'BI-RADS 1': 0, 'BI-RADS 2': 0, 'BI-RADS 3': 0, 'BI-RADS 4': 1, 'BI-RADS 5': 1})
            #class_name = self.mammography.iloc[idx, 7]
            #if class_name in ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3']:
            #   labels = 0.0
            #elif class_name in ['BI-RADS 4', 'BI-RADS 5']:
            #    labels = 1.0
            return labels
        elif self.task == 'BreastDensity4':
            labels = self.mammography['breast_density'].replace({'DENSITY A': 0, 'DENSITY B': 1 , 'DENSITY C': 2, 'DENSITY D': 3})
            #class_name = self.mammography.iloc[idx, 8]
            #if class_name in ['DENSITY A']:
            #    labels = 0
            #elif class_name in ['DENSITY B']:
            #    labels = 1
            #elif class_name in ['DENSITY C']:
            #    labels = 2
            #elif class_name in ['DENSITY D']:
            #    labels = 3
            return labels 
            
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.mammography.iat[idx,0],
                            self.mammography.iat[idx,2]+'.png')
        image = Image.open(img_path)
        #image = self.background_crop(image)
        # Assuming `image` is a single-channel grayscale image
        image_3c = Image.merge('RGB', (image, image, image))

        image_3c = image_3c.resize((self.image_height,self.image_width))

        # Get data for the image from the corresponding row in the CSV file
        study_id = self.mammography.iat[idx,0]
        laterality = self.mammography.iat[idx,3]
        view_position = self.mammography.iat[idx,4]
        breast_birads = self.mammography.iat[idx,7]
        breast_density = self.mammography.iat[idx,8]
        labels = self.class_name_to_labels()
        #print(labels.head())
        label = labels.iat[idx]
        #breast_density = label_encoder.transform([breast_density])[0]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image_3c = self.transform(image_3c)

        sample = {'image': image_3c, 'label': label, 'study_id': study_id, 'laterality':laterality, 'view_position':view_position, 'breast_birads': breast_birads, 'breast_density':breast_density}

        return sample
        
    def get_class_counts(self):
        class_counts = self.mammography['breast_density'].value_counts().sort_index()
        return class_counts.tolist()

    def get_labels(self):
        labels = self.mammography['breast_density'].tolist()
        return labels
    
    def get_weights(self):
        labels = self.class_name_to_labels()
        class_counts = labels.value_counts().sort_index()
        class_frequencies = class_counts / len(labels)
        class_weights = 1.0 / torch.tensor(class_frequencies, dtype=torch.float)
        class_weights /= class_weights.sum()  # Normalize the weights to have a sum of 1.0
        sample_weights = np.array([class_weights[label] for label in labels])
        sample_weights = torch.from_numpy(sample_weights)
        print(sample_weights.size())
        return sample_weights