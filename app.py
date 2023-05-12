import os
import webbrowser
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from flask import Flask, send_from_directory

app = Flask(__name__)

# configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# configure allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# load pre-trained model
model_path ='./modelENV.h5'
model = tf.keras.models.load_model(model_path)

# define function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# define function to predict tumor type
def predict_tumor_type(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    categories = ['Astrocitoma', 'Carcinoma', 'Ependimoma', 'Ganglioglioma', 'Germinoma', 'Glioblastoma', 'Granuloma', 'Meduloblastoma', 'Meningioma', 'Neurocitoma', 'Normal', 'Oligodendroglioma', 'Papiloma', 'Schwannoma', 'Tuberculoma']
    index = np.argmax(pred)
    prediction = categories[index]
    confidence_score = pred[0][index]
    percentage = round(confidence_score, 2) * 100
    return prediction , prediction, percentage
    
@app.route('/')
def index():
    return render_template('Brain_T.html')

@app.route('/predict', methods=['POST'])
def predict():
    # check if file was uploaded
    if 'file' not in request.files:
        return render_template('Brain_T.html', error='Please select an image.')

    file = request.files['file']

    # check if file was selected
    if file.filename == '':
        return render_template('Brain_T.html', error='Please select a new image.')

    # check if file is allowed
    if not allowed_file(file.filename):
        return render_template('Brain_T.html', error='Allowed file types are jpg, jpeg, and png.')

    # save file to upload folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # predict tumor type
    tumor_type, result_class, confidence_score = predict_tumor_type(file_path)

    return render_template('Brain_T.html', filename=filename, result=tumor_type, result_class=result_class, confidence_score=confidence_score)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # webbrowser.open_new('http://127.0.0.1:5000/')
    app.run(debug=True)
