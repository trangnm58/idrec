from django.conf.urls import url

from idocr.views import ImageToString

urlpatterns = [
    url(r'^image_to_string', ImageToString.as_view(), name='image_to_string'),
]