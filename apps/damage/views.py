from django.shortcuts import render
from django.contrib.messages.views import SuccessMessageMixin
from django.views.generic import ListView, CreateView
from .models import CarDamage, UploadCarDamage, UploadPicture
from .forms import UploadImageForm
from django.urls import reverse_lazy
from carDamage import settings
from django.http import HttpResponse


# car damage detection
import os
import json
import h5py
import numpy as np
import pickle as pk
# from PIL import Image
# keras imports
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
import tensorflow as tf


# Create your views here.

class IndexDamageView(SuccessMessageMixin, ListView):
    model = CarDamage
    template_name = 'index.html'


class UploadDamageView(SuccessMessageMixin, CreateView):
    model = UploadPicture
    form_class = UploadImageForm
    template_name = 'upload.html'
    success_message = 'Image Successfully Uploaded'
    success_url = reverse_lazy('carDamage:upload')

    def form_valid(self, form):
        image = form.save()
        image = settings.MEDIA_URL + image.file_name.name
        self.request.session['img_path'] = image
        return super(UploadDamageView, self).form_valid(form)


# car damage detection Code
# ************************* Prepare Image for processing ***********************
def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# Loading  valid categories for identifying cars using VGG16
with open('static/ml/cat_counter.pk', 'rb') as f:
    cat_counter = pk.load(f)

# shortlisting top 27 Categories that VGG16 stores for cars (Can be altered for less or more)
cat_list = [k for k, v in cat_counter.most_common()[:27]]

global graph
# graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph()


# ~~~~~~~~~~~~~~~ Prapare the flat image~~~~~~~~~~~~~
def prepare_flat(img_224):
    base_model = load_model('static/ml/vgg16.h5')
    model = Model(base_model.input, base_model.get_layer('fc1').output)
    feature = model.predict(img_224)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    return flat


# Models, Weights and Categories
# ******************************************************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~ FIRST Check- CAR OR NOT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ******************************************************************************

CLASS_INDEX_PATH = 'static/ml/imagenet_class_index.json'


def get_predictions(preds, top=5):
    global CLASS_INDEX
    CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def car_categories_check(img_224):
    first_check = load_model('static/ml/vgg16.h5')
    print("Validating that this is a picture of your car...")
    out = first_check.predict(img_224)
    top = get_predictions(out, top=5)
    for j in top[0]:
        if j[0:2] in cat_list:
            print("Car Check Passed!!!")
            print("\n")
            return True
    return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIRST check ENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~ SECOND CHECK - DAMAGED OR NOT~~~~~~~~~~~~~~~~~~~~~~~~

def car_damage_check(img_flat):
    second_check = pk.load(open('static/ml/second_check.pickle', 'rb'))  # damaged vs whole - trained model
    print("Validating that damage exists...")
    train_labels = ['00-damage', '01-whole']
    preds = second_check.predict(img_flat)
    prediction = train_labels[preds[0]]

    if train_labels[preds[0]] == '00-damage':
        print("Validation complete - proceeding to location and severity determination")
        print("\n")
        return True
    else:
        return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SECOND CHECK ENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ THIRD CHECK - Location and Severity Assesment~~~~~~~~~~~~~

def location_assessment(img_flat):
    print("Validating the damage area - Front, Rear or Side")
    third_check = pk.load(open("static/ml/third_check.pickle", 'rb'))
    train_labels = ['Front', 'Rear', 'Side']
    preds = third_check.predict(img_flat)
    prediction = train_labels[preds[0]]
    print("Your Car is damaged at - " + train_labels[preds[0]])
    print("Location assesment complete")
    print("\n")
    return prediction


def severity_assessment(img_flat):
    print("Validating the Severity...")
    fourth_check = pk.load(open("static/ml/fourth_check.pickle", 'rb'))
    train_labels = ['Minor', 'Moderate', 'Severe']
    preds = fourth_check.predict(img_flat)
    prediction = train_labels[preds[0]]
    print("Your Car damage impact is - " + train_labels[preds[0]])
    print("Severity assesment complete")
    print("\n")
    print("Thank you for using our service")
    print("More such kits in pipeline")
    return prediction


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ THIRD CHECK ENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ENGINE  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# load models

import pathlib


def engine(request):
    MyCar = request.session['img_path']
    # img_path = 'http://127.0.0.1:8000' + MyCar
    # img_path = MyCar.replace('/media/', '')
    img_path = '/Users/bikashsubedi/dev/AI-ML-DS/carDocScanML/damage-car-detecting-django-app' + MyCar


    request.session.pop('img_path', None)
    request.session.modified = True

    with graph.as_default():

        img_224 = prepare_img_224(img_path)
        img_flat = prepare_flat(img_224)
        g1 = car_categories_check(img_224)
        g2 = car_damage_check(img_flat)

        while True:
            try:

                if g1 is False:
                    g1_pic = "Looks Like you are testing with random image, please choose image of car"
                    g2_pic = 'N/A'
                    g3 = 'N/A'
                    g4 = 'N/A'
                    ns = 'N/A'
                    break
                else:
                    g1_pic = "Its a Car"

                if g2 is False:
                    g2_pic = "Are you sure your car is damaged?. Make sure you click a clear picture of your car"
                    g3 = 'N/A'
                    g4 = 'N/A'
                    ns = 'N/A'
                    break
                else:
                    g2_pic = "Car Damaged. Refer below sections for Location and Severity"

                    g3 = location_assessment(img_flat)
                    g4 = severity_assessment(img_flat)
                    ns = 'a). Create a report and send to Vendor \n b). Proceed to cost estimation \n c). Estimate TAT'
                    break

            except:
                break

    src = '/Users/bikashsubedi/dev/AI-ML-DS/carDocScanML/damage-car-detecting-django-app/media/upload_car/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".jpg"):
            os.remove(src + image_file_name)

    K.clear_session()

    context = {'g1_pic': g1_pic, 'g2_pic': g2_pic, 'loc': g3}

    results = json.dumps(context)
    return HttpResponse(results, content_type='application/json')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ENGINE ENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ******************************* KYC Begins ***********************************

def kyc_pred(img_flat):
    print("Identifying the document type")
    kyc_check = pk.load(open("static/ml/kyc.pickle", 'rb'))
    prob = kyc_check.predict_proba(img_flat)
    probadhaar = (prob[0][0]) * 100
    probpan = (prob[0][1]) * 100

    if probpan >= 98:
        pred_kyc = "Its a pan card"
    elif probadhaar >= 98:
        pred_kyc = "Its an adhaar card"
    else:
        pred_kyc = "Neither pan nor adhaar"

    return pred_kyc


def pan_adhaar(request):
    img_path = request.session['image_path10']
    request.session.pop('image_path10', None)
    request.session.modified = True
    with graph.as_default():
        img_224 = prepare_img_224(img_path)
        img_flat = prepare_flat(img_224)
        kyc = kyc_pred(img_flat)
        print(kyc)

    src = 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        os.remove(src + image_file_name)

    K.clear_session()

    context10 = {'kyc': kyc}

    result_kyc = json.dumps(context10)
    return HttpResponse(result_kyc, content_type='application/json')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ KYC ENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
