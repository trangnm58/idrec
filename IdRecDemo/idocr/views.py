from __future__ import division, print_function, unicode_literals
import os
import cv2
import numpy as np
from base64 import b64encode
from keras.backend import clear_session
from django.views import View
from django.shortcuts import render
from django.conf import settings

from idocr.segmentation.segmentor import Segmentor
from idocr.recognition.recognizer import Recognizer

SYSTEM_TESTSET = os.path.join(settings.PROJECT_ROOT, 'idocr/all_data/system_testset/')

class ImageToString(View):
    def get(self, request):
        filename = os.path.join(SYSTEM_TESTSET, os.listdir(SYSTEM_TESTSET)[0])
        demo_image = cv2.imread(filename)
        string = cv2.imencode('.jpg', demo_image)[1]
        base64_str = b64encode(string)
        
        mime = "image/jpg"
        mime = mime + ";" if mime else ";"
        image = "data:{}base64,{}".format(mime, base64_str.decode("utf-8"))
        
        return render(request, "idocr/index.html", {'image': image})
    
    def post(self, request):
        raw_image = request.FILES.get("image")
        if raw_image:
            raw_image = raw_image.read()
            img = cv2.imdecode(np.fromstring(raw_image, np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            filename = os.path.join(SYSTEM_TESTSET, os.listdir(SYSTEM_TESTSET)[0])
            img = cv2.imread(filename)
            
        segmentor = Segmentor(os.path.join(settings.PROJECT_ROOT, "idocr/segmentation/"))
        image = segmentor.segment_as_predict(img)  # Image object
        segmentor.refine_segments(image)  # image's segmentation is refined
        
        recognizer = Recognizer(os.path.join(settings.PROJECT_ROOT, "idocr/recognition/"))
        recognizer.recognize_as_predict(image)
        recognizer.post_process(image)
        
        clear_session()
        
        text = '\n\n'.join([f.postprocessed_text for f in image.fields])

        string = cv2.imencode('.jpg', image.image)[1]
        base64_str = b64encode(string)
        
        mime = "image/jpg"
        mime = mime + ";" if mime else ";"
        b64image = "data:{}base64,{}".format(mime, base64_str.decode("utf-8"))

        return render(request, "idocr/index.html", {'text': text, 'image': b64image})
