import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import datetime

class UnetBrainTumorSegmentation:
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
        model = keras.models.load_model(model_path, compile=False)
        X = self.preprocess_nii_file(input_filepath)
        predictions = model.predict(X, verbose=1)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_filepath = f'{input_filepath}_unet_detected_slice_num_{timestamp}.png'
        output_filepath = output_filepath.replace("uploaded", "generated")
        output_filepath = output_filepath.replace("preprocessed", "generated")
        output_filepaths = []
        
        for i, slice_num in enumerate(self.SELECTED_SLICES):
            plt.figure(figsize=(15, 10))
            for idx, (title, channel) in enumerate([('Original', 0), ('Necrotic/Core', 1), ('Edema', 2), ('Enhancing', 3), ('Combined', None)]):
                plt.subplot(1, 5, idx + 1)
                plt.imshow(X[i, :, :, 0], cmap='gray')
                if title != 'Original':
                    if title == 'Combined':
                        overlay = np.zeros((*predictions[i, :, :, 1].shape, 3))
                        overlay[:, :, 0] = predictions[i, :, :, 1]  # Red - Necrotic
                        overlay[:, :, 1] = predictions[i, :, :, 3]  # Green - Enhancing
                        overlay[:, :, 2] = predictions[i, :, :, 2]  # Blue - Edema
                        plt.imshow(overlay, alpha=0.5)
                    else:
                        plt.imshow(predictions[i, :, :, channel], alpha=0.5, cmap='Reds' if channel == 1 else 'Blues' if channel == 2 else 'Greens')
                    plt.title(title)
                else:
                    plt.title(f'Original slice {slice_num}')
                plt.axis('off')
            figpath = output_filepath.replace("slice_num", str(slice_num))
            plt.savefig(figpath)
            print(f'Output filepath for slice {slice_num} is: {figpath}')
            output_filepaths.append(figpath)
            # plt.show()
        return output_filepaths
