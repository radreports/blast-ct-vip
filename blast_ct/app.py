from flask import Flask, request, send_file, render_template
import torch,json
import nibabel as nib
import numpy as np
import os
import uuid
from blast_ct.read_config import get_model, get_test_loader



app = Flask(__name__)

# Load the pre-trained model
# model = torch.load('data/saved_models/your_model.pt')
config_file = 'data/config.json'
with open(config_file, 'r') as f:
        config = json.load(f)
model = model = get_model(config)
model.eval()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/infer', methods=['POST'])
def infer():
    # Save uploaded file
    file = request.files['file']
    file_extension = os.path.splitext(file.filename)[1]
    
    if file_extension not in ['.nii', '.nii.gz']:
        return "Invalid file format. Please upload a .nii or .nii.gz file.", 400

    input_path = os.path.join('/tmp', str(uuid.uuid4()) + file_extension)
    file.save(input_path)

    # Decompress if .nii.gz
    if file_extension == '.nii.gz':
        with gzip.open(input_path, 'rb') as f_in:
            decompressed_path = input_path.replace('.nii.gz', '.nii')
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        input_path = decompressed_path

    # Load the input image
    img = nib.load(input_path)
    img_data = img.get_fdata()
    img_data = np.expand_dims(img_data, axis=(0, 1))  # Add batch and channel dimensions

    # Convert to tensor
    image = torch.from_numpy(img_data).float()

    # Run inference
    with torch.no_grad():
        outputs = model(image)
    
    # Convert output to numpy
    output_image = outputs[0].squeeze().cpu().numpy()  # Remove batch and channel dimensions

    # Save the output image
    output_path = os.path.join('/tmp', str(uuid.uuid4()) + '_output.nii.gz')
    output_nifti = nib.Nifti1Image(output_image, img.affine)
    nib.save(output_nifti, output_path)

    return send_file(output_path, as_attachment=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
