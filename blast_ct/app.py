from flask import Flask, request, send_file, render_template
import subprocess
import os
import uuid

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def infer():
    # Save uploaded file
    file = request.files['file']
    file_extension = os.path.splitext(file.filename)[1]
    print("file_extension: ", file_extension)
    print("Input file is: ", file)
    # if file_extension not in ['.nii', '.nii.gz']:
    #     return "Invalid file format. Please upload a .nii or .nii.gz file.", 400

    input_path = os.path.join('/tmp', str(uuid.uuid4()) + file_extension)
    file.save(input_path)

    # Generate unique output path
    output_path = os.path.join('/tmp', str(uuid.uuid4()) + '_output.nii.gz')

    # Execute the command-line tool
    command = ['blast-ct', '--input', input_path, '--output', output_path]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error during inference: {e}", 500

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
