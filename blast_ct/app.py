from flask import Flask, request, send_file
import torch
import nibabel as nib
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load the pre-trained model
model = torch.load('data/saved_models/your_model.pt')
model.eval()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/infer', methods=['POST'])
def infer():
    # Save uploaded file
    file = request.files['file']
    input_path = os.path.join('/tmp', str(uuid.uuid4()) + '.nii.gz')
    file.save(input_path)

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
