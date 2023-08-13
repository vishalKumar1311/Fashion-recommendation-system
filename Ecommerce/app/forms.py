from .models import Profile
from django import forms
from django.db import models
from django.forms import ModelForm

class Pimg(ModelForm):
   class Meta:
      model=Profile
      fields=['image']
