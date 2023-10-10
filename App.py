import base64
from flask import Flask, request, json,jsonify
import os
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pydicom
from PIL import Image
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
from sklearn.metrics import confusion_matrix


my_awesome_app = Flask(__name__)
CORS(my_awesome_app)
my_awesome_app.secret_key = "caircocoders-ednalan"
UPLOAD_FOLDER = 'static/uploads'
my_awesome_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# my_awesome_app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_CLASIFER = 'static/clasifer'
my_awesome_app.config['UPLOAD_CLASIFER'] = UPLOAD_CLASIFER
ALLOWED_EXTENSIONS = set(['dcm'])

@my_awesome_app.route('/')
def hello_world():
    resp = jsonify({'message': 'Ping'})
    resp.status_code = 201
    return resp

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#tempat untuk request file(mengambil)
@my_awesome_app.route('/upload', methods=['POST'])
def upload_file():
    filepath = ""
    files = request.files.getlist('files')
    url = request.form.get('files')
    print(files)
    print(url)
    errors = {}
    success = False
    if (url == None):
        print("file")
        for file in files:
            if file and allowed_file(file.filename):
                print("dajndaw")
                filename = secure_filename(file.filename)
                filepath = os.path.join(my_awesome_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                success = True
                res = proses(filepath)
                resp = jsonify(res)
                resp.status_code = 201
                return resp
            else:
                errors['message'] = 'File type is not allowed'
                resp = jsonify(res)
                resp.status_code = 500
                return resp
    else:
        print("hbhbbbd")
        filepath = os.path.join(my_awesome_app.config['UPLOAD_FOLDER'], url)
        success = True
        res = proses(filepath)
        resp = jsonify(res)
        resp.status_code = 201
        return resp
    # if success and errors:
    #     errors['message'] = 'File(s) successfully uploaded'
    #     resp = jsonify(errors)
    #     resp.status_code = 500
    #     return resp
    # if success:
    #     resp = jsonify({'message': 'Files successfully uploaded'})
    #     resp.status_code = 201
    #     return resp
    # else:
    #     resp = jsonify(errors)
    #     resp.status_code = 500
    #     return resp

def proses(filepath):
    # dicom ke jpg
    # membaca file DICOM
    dicom_file = pydicom.dcmread(filepath)
    # membuat gambar PIL dari data DICOM
    imagedcm = Image.fromarray(dicom_file.pixel_array)
    # menyimpan gambar sebagai JPG
    imagedcm.save("static/clasifer/clasifier.jpg")

    # ----klasifikasi----
    kelas = ["stadium 1", "stadium 2", "stadium 3", "stadium 4"]
    test = []
    # dirimg = final_image.save(os.path.join(my_awesome_app.config['UPLOAD_CLASIFER'], "clasifier.jpg"))
    dirimg = os.path.join("static/clasifer/clasifier.jpg")
    print("dir = ", dirimg)
    kanker_img = cv2.imread(dirimg, 0)
    kanker_img = cv2.resize(kanker_img, (50, 50))
    image = np.array(kanker_img).flatten()
    test.append(image)
    print("image = ", image)
    print("test= ", test)

    #training testing #
    data = []
    dir= "E:\\dataset deep learning\\Stadium kanker payudara\\Dataset\\new\\Stadium kanker payudara\\"
    # kelas = ["stadium 4", "stadium 3", "stadium 2", "stadium 1"]
    for category in kelas:
        #dimasukkan kedalam path
      path = os.path.join(dir,category)
      label = kelas.index(category)
      for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        kanker_img=cv2.imread(imgpath,0)
        try:
          kanker_img=cv2.resize(kanker_img,(50,50))
          image = np.array(kanker_img).flatten()

          data.append([image,label])
        except Exception as e:
          pass

    pick_in = open('data.pickle','wb')
    pickle.dump(data,pick_in)
    pick_in.close ()

    pick_in = open('data.pickle', 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    random.shuffle(data)
    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.20, stratify=labels)
    model_rbf = SVC(C=50, kernel='rbf', gamma="scale")
    model_rbf.fit(xtrain, ytrain)

    model_poly = SVC(C=50, kernel='poly', gamma="scale")
    model_poly.fit(xtrain, ytrain)

    #-------------------------------------

    # confusion matrix
    y_pred_rbf = model_rbf.predict(xtest)
    y_pred_poly = model_poly.predict(xtest)

    # membuat confusion matrix
    cm_rbf = confusion_matrix(ytest, y_pred_rbf)
    print("Confusion Matrix RBF:")
    print(cm_rbf)

    cm_poly = confusion_matrix(ytest, y_pred_poly)
    print("Confusion Matrix poly:")
    print(cm_poly)

    print("data = ", len(data))
    print("training = ", len(xtrain))
    print("testing = ", len(xtest))
    print("p = ", features[0])

    prediction_rbf = model_rbf.predict(test)
    predicted_category_rbf = kelas[prediction_rbf[0]]

    prediction_poly = model_poly.predict(test)
    predicted_category_poly = kelas[prediction_poly[0]]

    accuracy_rbf = model_rbf.score(features, labels)
    accuracy_poly = model_poly.score(features, labels)

    print('Accuracy RBF: ', accuracy_rbf)
    print('Accuracy Polynomial: ', accuracy_poly)

    print('Prediksi RBF: ', predicted_category_rbf)
    print('Prediksi Polynomial: ', predicted_category_poly)
    return {"akurasi_rbf": accuracy_rbf, "prediksi_rbf":predicted_category_rbf,"akurasi_poly": accuracy_poly, "prediksi_poly":predicted_category_poly}

@my_awesome_app.route('/getfile')
def getfile():
    with open("kanker.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    print("imags", image_file)
    return image_file


if __name__ == '__main__':
    my_awesome_app.run()