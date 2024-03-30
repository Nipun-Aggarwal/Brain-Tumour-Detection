import os
import nibabel as nib
from nilearn import image as nli
from nilearn.masking import compute_brain_mask
import numpy as np
import datetime

class MRI_Pre_Processor:
    def __init__(self, input_path):
        self.input_path = input_path

    def preprocess_and_save(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        processed_filepath = f'{self.input_path}_processed_{timestamp}.nii'
        processed_filepath = processed_filepath.replace("uploaded", "preprocessed")

        # Load the NIfTI file
        nii_img = nib.load(self.input_path)

        # Interpolate to 1mm³ resolution
        target_affine = np.eye(3)  # Target resolution set to 1mm x 1mm x 1mm
        resampled_img = nli.resample_img(nii_img, target_affine=target_affine, interpolation='nearest')

        # Skull stripping
        brain_mask = compute_brain_mask(resampled_img)
        brain_img = nli.math_img('img * mask', img=resampled_img, mask=brain_mask)

        # Save the preprocessed image in uncompressed .nii format
        nib.save(brain_img, processed_filepath)
        print(f'Preprocessed file saved as {processed_filepath}')
        return processed_filepath

# def main():
#     input_file = "path/to/your/input/brain.nii"  # Update this to your file path
#     output_dir = "path/to/your/output/directory"  # Update this to your desired output directory
    
#     processor = MRI_Processor(input_file, output_dir)
#     processor.preprocess_and_save()

# if __name__ == "__main__":
#     main()