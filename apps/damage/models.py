from django.db import models


# Create your models here.

class CarDamage(models.Model):
    pass


class UploadCarDamage(models.Model):
    pass


class UploadPicture(models.Model):
    file_name = models.ImageField(blank=True, upload_to='upload_car', verbose_name='Image Name')
    created_at = models.DateTimeField(auto_now_add=True)
