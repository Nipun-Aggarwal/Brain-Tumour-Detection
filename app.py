from flask import Flask, render_template, request, redirect, url_for
import os
from webapp.components.preprocess.preprocess import MRI_Pre_Processor
from webapp.components.unet.UnetPred import UnetBrainTumorSegmentation
from webapp.components.resnet.ResNetPred import ResnetBrainTumorSegmentation

app = Flask(__name__)
UPLOAD_FOLDER = 'static/images/uploaded'
DISPLAY_FOLDER = 'images/generated'
MODEL_FOLDER = 'webapp/models/'
UPLOAD_DISPLAY_FOLDER = 'images/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DISPLAY_FOLDER'] = DISPLAY_FOLDER
app.config['UPLOAD_DISPLAY_FOLDER'] = UPLOAD_DISPLAY_FOLDER


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/uploadImage', methods=['POST'])
def uploadImage():
    # print('Inside Uploading image')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('upload', err_msg='No file part'))

        file = request.files['file']

        # Check if the file name is empty
        if file.filename == '':
            return redirect(url_for('upload', err_msg='No file selected'))

        # Check if the file extension is .nii
        if file.filename.split('.')[-1].lower() != 'nii':
            # return 'Please upload a .nii file'
            return redirect(url_for('upload', err_msg='Please upload a .nii file'))

        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename).replace('\\', '/')
        file.save(filepath)

        # Check if preprocessing checkbox is checked
        preprocess = False
        if 'preprocessCheckbox' in request.form:
            preprocess = True
        
        print(f'Preprocess value is {preprocess}')

        if(preprocess):
            preprocessor = MRI_Pre_Processor(filepath)
            filepath = preprocessor.preprocess_and_save()

        # Retrieve the model selection from the form
        selected_model = request.form.get('model', 'UNET')
        print(f'Selected model: {selected_model}')

        output_filepaths = []

        if selected_model == 'UNETWITHRESNET':
            model_path = MODEL_FOLDER + 'RESNET_MODEL.h5'
            detector = ResnetBrainTumorSegmentation()

        if selected_model == 'UNET':
            model_path = MODEL_FOLDER + 'UNET_MODEL.h5'
            detector = UnetBrainTumorSegmentation()        
        
        output_filepaths = detector.predict_and_show(filepath, model_path)

        if len(output_filepaths) >= 4:
            slice_31_filename =  os.path.basename(output_filepaths[0])
            slice_62_filename =  os.path.basename(output_filepaths[1])
            slice_93_filename =  os.path.basename(output_filepaths[2])
            slice_124_filename =  os.path.basename(output_filepaths[3])
            return redirect(url_for('display',
                                    slice_31_filename=slice_31_filename,
                                    slice_62_filename=slice_62_filename,
                                    slice_93_filename=slice_93_filename,
                                    slice_124_filename=slice_124_filename))

        return redirect(url_for('upload', err_msg='Error in Image Generation'))
    return redirect(url_for('upload', err_msg='No Post Method'))


@app.route('/display')
def display():
    
    slice_31_filename = request.args.get('slice_31_filename', default=None)
    slice_62_filename = request.args.get('slice_62_filename', default=None)
    slice_93_filename = request.args.get('slice_93_filename', default=None)
    slice_124_filename = request.args.get('slice_124_filename', default=None)

    slice_31_image_url = os.path.join(app.config['DISPLAY_FOLDER'], slice_31_filename).replace('\\', '/')
    slice_62_image_url = os.path.join(app.config['DISPLAY_FOLDER'], slice_62_filename).replace('\\', '/')
    slice_93_image_url = os.path.join(app.config['DISPLAY_FOLDER'], slice_93_filename).replace('\\', '/')
    slice_124_image_url = os.path.join(app.config['DISPLAY_FOLDER'], slice_124_filename).replace('\\', '/')


    return render_template('display.html',
                           slice_31_image_url=slice_31_image_url,
                           slice_62_image_url=slice_62_image_url,
                           slice_93_image_url=slice_93_image_url,
                           slice_124_image_url=slice_124_image_url)


if __name__ == '__main__':
    app.run(debug=True)
