from django.urls import path
from .views import IndexDamageView, UploadDamageView, engine


app_name = 'carDamage'
urlpatterns = [
    path('', IndexDamageView.as_view(), name='home'),
    path('upload/', UploadDamageView.as_view(), name='upload'),
    path('process/', engine, name='engine'),
]
