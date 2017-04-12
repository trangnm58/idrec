from __future__ import division, print_function, unicode_literals
import os
import zipfile
import cv2
import numpy as np
from base64 import b64encode
from keras.backend import clear_session
from django.views import View
from ratelimit.mixins import RatelimitMixin
from django.shortcuts import render
from django.conf import settings

from idocr.segmentation.segmentor import Segmentor
from idocr.recognition.recognizer import Recognizer


class ImageToString(RatelimitMixin, View):
    ratelimit_key = 'ip'
    ratelimit_rate = '10/3000d'
    ratelimit_block = True
    ratelimit_method = 'POST'
    
    def get(self, request):
        return render(request, "idocr/index.html")
    
    def post(self, request):
        file = request.FILES.get("image")

        if file:
            if file.size > 100 * 500000:
                # File is too large
                return render(request, "idocr/index.html")
            
            all_imgs = []  # data image to return
            all_text = []
            
            clear_session()
            self.segmentor = Segmentor(os.path.join(settings.PROJECT_ROOT, "idocr/segmentation/"))
            self.recognizer = Recognizer(os.path.join(settings.PROJECT_ROOT, "idocr/recognition/"))
            
            if zipfile.is_zipfile(file):
                zipped_imgs = zipfile.ZipFile(file)
                name_list = zipped_imgs.namelist()
                
                for n in name_list:
                    data = zipped_imgs.read(n)
                    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)
                
                    b64image, text = self._process_image(img)
                    all_text.append(text)
                    all_imgs.append(b64image)

            else:
                # process a single image
                file.seek(0)
                raw_image = file.read()
                img = cv2.imdecode(np.fromstring(raw_image, np.uint8), cv2.IMREAD_UNCHANGED)

                b64image, text = self._process_image(img)
                all_text.append(text)
                all_imgs.append(b64image)

            clear_session()

            res = []
            for i in range(len(all_imgs)):
                res.append({"text": all_text[i], "image": all_imgs[i]})
                
            return render(request, "idocr/index.html", {'result': res})
        else:
            return render(request, "idocr/index.html")
        
    def _process_image(self, img):
        image = self.segmentor.segment_as_predict(img)  # Image object
        
        self.segmentor.refine_segments(image)  # image's segmentation is refined
        
        self.recognizer.recognize_as_predict(image)
        self.recognizer.post_process(image)
        
        text = '\n\n'.join([f.postprocessed_text for f in image.fields])
        
        string = cv2.imencode('.jpg', image.image)[1]
        base64_str = b64encode(string)
        
        mime = "image/jpg"
        mime = mime + ";" if mime else ";"
        b64image = "data:{}base64,{}".format(mime, base64_str.decode("utf-8"))

        return b64image, text
