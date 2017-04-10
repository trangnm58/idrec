from django.conf.urls import url, include

urlpatterns = [
    url(r'^idocr/', include('idocr.urls', namespace="idocr")),
]
