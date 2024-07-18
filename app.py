from flask import Flask, request, render_template, send_file, send_from_directory
import os
import subprocess
import pandas as pd

app = Flask(__name__)

# 文件上傳目錄
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'pdf_files' not in request.files or 'question_file' not in request.files:
        return "No file part", 400

    pdf_files = request.files.getlist('pdf_files')
    question_file = request.files['question_file']

    pdf_paths = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
        pdf_file.save(pdf_path)
        pdf_paths.append(pdf_path)

    question_path = os.path.join(app.config['UPLOAD_FOLDER'], question_file.filename)
    question_file.save(question_path)

    # 調用 mistral-7B-instruct.py
    mistral_command = f"python mistral-7B-instruct.py {question_path} {' '.join(pdf_paths)}"
    subprocess.run(mistral_command, shell=True)

    # 調用 gemini.py
    gemini_command = "python gemini.py"
    subprocess.run(gemini_command, shell=True)

    return render_template('results.html')

@app.route('/download')
def download_file():
    path = "gemini_responses.txt"
    return send_file(path, as_attachment=True)

@app.route('/hammett')
def hammett():
    return render_template('hammett_plot.html')

@app.route('/plot', methods=['POST'])
def plot():
    from hammett_plot import generate_hammett_plot

    y_axis_label = request.form['y_axis_label']
    log_transform = request.form['log_transform'] == 'true'
    
    substituents = []
    values = []
    for key in request.form:
        if key.startswith('substituent'):
            substituents.append(request.form[key])
        elif key.startswith('value'):
            values.append(request.form[key])
    
    image_path, output_path = generate_hammett_plot(substituents, values, y_axis_label, log_transform, app.config['UPLOAD_FOLDER'])
    
    return render_template('plot_result.html', image_path=image_path, output_path=output_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
