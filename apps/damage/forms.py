from django import forms
from .models import UploadPicture


class UploadImageForm(forms.ModelForm):
    class Meta:
        model = UploadPicture
        fields = ['file_name']
