from django.contrib import admin
from .models import Product,Category,Order,Size,Profile,Fairtone,Darktone,Mediumtone
# Register your models here.

admin.site.register(Category)
admin.site.register(Product)
admin.site.register(Fairtone)
admin.site.register(Mediumtone)
admin.site.register(Darktone)
admin.site.register(Order)
admin.site.register(Size)
admin.site.register(Profile)
