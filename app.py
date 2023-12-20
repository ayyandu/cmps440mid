from flask import Flask, render_template, request, redirect, send_file
import os
import qrcode
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from zxing import BarCodeReader
from io import BytesIO
from roboflow import Roboflow
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}



rf = Roboflow(api_key="NgFh3N0J8nQaFbfhEMcR")
project = rf.workspace().project("card-z6tto")
model = project.version(1).model

project_2 = rf.workspace().project("ambulance-detection-6vckd")
model_2 = project_2.version(1).model



signal_state = "red"

def check_for_ambulance(image_path):
    global model_2

    # Use Roboflow model for inference
    result = model_2.predict(image_path, confidence=40, overlap=30).json()

    # Check if the prediction contains the "ambulance" class
    for prediction in result['predictions']:
        if prediction['class'] == 'ambulance':
            return True

    return False

@app.route('/signal', methods=['GET', 'POST'])
def traffic():
    global signal_state
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(io.BytesIO(file.read()))
            img_path = 'temp_image.jpg'  # Save the image temporarily
            img.save(img_path)

            # Check for ambulance in the uploaded image
            ambulance_detected = check_for_ambulance(img_path)

            if ambulance_detected:
                signal_state = "green"
            else:
                signal_state = "red"

    return render_template('traffic.html', signal=f'/static/{signal_state}.png')

# Function to perform inference
def perform_inference(image_path):
    result = model.predict(image_path, confidence=40, overlap=30).json()
    return result

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def homepage():
    return render_template('main.html')

@app.route('/u')
def index():
    return render_template('index.html')

@app.route('/card')
def cardhome():
    return render_template('cardupload.html')

@app.route('/traffic')
def traffichome():
    return render_template('traffic.html')



@app.route('/cardprocess', methods=['POST'])
def card_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = perform_inference(file_path)
            return render_template('cardresult.html', image_file=filename, result=result)
    return render_template('cardupload.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('result.html', filename=filename)
    else:
        return redirect(request.url)

@app.route('/process_image', methods=['POST'])
def process_image():
    filename = request.form['filename']
    action = request.form['action']
    processed_image = None

    if action == 'blur':
        processed_image = apply_blur(filename)
    elif action == 'sharpen':
        processed_image = apply_sharpen(filename)
    elif action == 'emboss':
        processed_image = apply_emboss(filename)
    else:
        # 'original' or unknown action, show original image
        return render_template('result.html', filename=filename)

    return render_template('result.html', filename=filename, processed_image=processed_image)

def apply_blur(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    processed_image_path = f"blurred_{filename}"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], processed_image_path), blurred_image)
    return processed_image_path

def apply_sharpen(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    processed_image_path = f"sharpened_{filename}"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], processed_image_path), sharpened_image)
    return processed_image_path

def apply_emboss(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath, 0)
    emboss_kernel = np.array([[0, -1, -1],
                              [1, 0, -1],
                              [1, 1, 0]])
    embossed_image = cv2.filter2D(image, -1, emboss_kernel)
    processed_image_path = f"embossed_{filename}"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], processed_image_path), embossed_image)
    return processed_image_path

tags = {}


@app.route('/toll')
def toll():
    return render_template('toll.html')

@app.route('/generate_tag', methods=['GET', 'POST'])
def generate_tag():
    tag_number = None
    if request.method == 'POST':
        tag_number = request.form['tag_number']
        initial_balance = float(request.form['initial_balance'])
        tags[tag_number] = initial_balance

    return render_template('generate_tag.html', tag_number=tag_number)

@app.route('/get_qr/<tag_number>')
def get_qr(tag_number):
    if tag_number in tags:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(tag_number)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    else:
        return "Tag not found"


@app.route('/qr_scanner')
def qr_scanner():
    return render_template('qr_scanner.html')

@app.route('/qr_upload')
def qr_upload():
    return render_template('qr_upload.html')

@app.route('/balance')
def balance():
    return render_template('balance.html')

@app.route('/process_qr', methods=['POST'])
def process_qr():
    if 'qr_code' in request.files:
        qr_file = request.files['qr_code']

        temp_file_path = 'temp_qr_code.png'
        qr_file.save(temp_file_path)

        reader = BarCodeReader()
        decoded_data = reader.decode(temp_file_path)

        os.remove(temp_file_path)

        if decoded_data:
            tag_number = decoded_data.parsed
            if tag_number in tags:
                if tags[tag_number] >= 5:
                    tags[tag_number] -= 5
                    new_balance = tags[tag_number]
                    message = f"$5 deducted. New balance for tag {tag_number} is ${new_balance}"
                    success = True
                else:
                    message = "Insufficient balance"
                    success = False
            else:
                message = "Tag not found"
                success = False
        else:
            message = "Could not decode the QR code"
            success = False

        # Render the result.html template with the processing outcome
        return render_template('tollresult.html', message=message, success=success)

    return "No QR code uploaded"

@app.route('/process_balance', methods=['POST'])
def process_balance():
    if 'qr_code' in request.files:
        qr_file = request.files['qr_code']

        temp_file_path = 'temp_qr_code.png'
        qr_file.save(temp_file_path)

        reader = BarCodeReader()
        decoded_data = reader.decode(temp_file_path)

        os.remove(temp_file_path)

        if decoded_data:
            tag_number = decoded_data.parsed
            if tag_number in tags:
                tag_balance = tags[tag_number]  # Get tag balance
                message = f"Tag {tag_number} has a balance of ${tag_balance}"
                success = True
            else:
                message = "Tag not found"
                success = False
        else:
            message = "Could not decode the QR code"
            success = False

        return render_template('showbalance.html', message=message, success=success)

    return "No QR code uploaded"



@app.route('/deduct_balance', methods=['POST'])
def deduct_balance():
    tag_number = request.form['tag_number']
    if tag_number in tags:
        if tags[tag_number] >= 5:
            tags[tag_number] -= 5
            new_balance = tags[tag_number]
            return f"$5 deducted. New balance for tag {tag_number} is ${new_balance}"
        else:
            return "Insufficient balance"
    else:
        return "Tag not found"

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/add_balance', methods=['POST'])
def add_balance():
    tag_number = request.form['tag_number']
    balance_to_add = float(request.form['balance_to_add'])
    if tag_number in tags:
        tags[tag_number] += balance_to_add
        new_balance = tags[tag_number]
        return f"${balance_to_add} added. New balance for tag {tag_number} is ${new_balance}"
    else:
        return "Tag not found"


if __name__ == '__main__':
    app.run(debug=False)