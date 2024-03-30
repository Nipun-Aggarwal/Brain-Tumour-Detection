import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import datetime

class ResnetBrainTumorSegmentation:
    def __init__(self):
        self.IMG_SIZE = 128
        self.VOLUME_SLICES = 155
        self.VOLUME_START_AT = 22
        self.SELECTED_SLICES = [31, 62, 93, 124]
        self.SEGMENT_CLASSES = {
            0: 'NOT tumor',
            1: 'NECROTIC/CORE',
            2: 'EDEMA',
            3: 'ENHANCING'
        }
        
    def preprocess_nii_file(self, nii_path):
        image = nib.load(nii_path).get_fdata()
        X = np.zeros((len(self.SELECTED_SLICES), self.IMG_SIZE, self.IMG_SIZE, 2))
        for i, j in enumerate(self.SELECTED_SLICES):
            X[i, :, :, 0] = cv2.resize(image[:, :, j], (self.IMG_SIZE, self.IMG_SIZE))
            X[i, :, :, 1] = cv2.resize(image[:, :, j], (self.IMG_SIZE, self.IMG_SIZE))
        X /= np.max(X)
        return X
    
    def predict_and_show(self, input_filepath, model_path):
        print("Inside Resnet", input_filepath)
        model = keras.models.load_model(model_path, compile=False)
        X = self.preprocess_nii_file(input_filepath)
        predictions = model.predict(X, verbose=1)
        print("Prediction done")
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_filepath = f'{input_filepath}_resnet_detected_slice_num_{timestamp}.png'
        print(f'Output filepath template: {output_filepath}')
        output_filepath = output_filepath.replace("uploaded", "generated")
        output_filepath = output_filepath.replace("preprocessed", "generated")
        output_filepaths = []
        
        for i, slice_num in enumerate(self.SELECTED_SLICES):
            plt.figure(figsize=(15, 10))
            plt.subplot(1, 5, 1)
            plt.imshow(X[i, :, :, 0], cmap='gray')
            plt.title(f'Original slice {slice_num}')
            plt.axis('off')

            for idx, (label, cmap_color) in enumerate([('Necrotic/Core', 'Reds'), ('Edema', 'Blues'), ('Enhancing', 'Greens')]):
                plt.subplot(1, 5, idx + 2)
                plt.imshow(X[i, :, :, 0], cmap='gray')
                plt.imshow(predictions[i, :, :, idx + 1], alpha=0.5, cmap=cmap_color)
                plt.title(label)
                plt.axis('off')

            plt.subplot(1, 5, 5)
            plt.imshow(X[i, :, :, 0], cmap='gray')
            overlay = np.zeros((*predictions[i, :, :, 1].shape, 3))
            overlay[..., 0] = predictions[i, ..., 1]  # Red channel - Necrotic
            overlay[..., 1] = predictions[i, ..., 3]  # Green channel - Enhancing
            overlay[..., 2] = predictions[i, ..., 2]  # Blue channel - Edema
            plt.imshow(overlay, alpha=0.5)
            plt.title('Combined')
            plt.axis('off')
            figpath = output_filepath.replace("slice_num", str(slice_num))
            plt.savefig(figpath)
            print(f'Output filepath for slice {slice_num} is: {figpath}')
            output_filepaths.append(figpath)
            # plt.show()
        return output_filepaths
