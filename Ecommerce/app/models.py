from django.db import models
from django.contrib.auth.models import User
# Create your models here.


class Size(models.Model):
   size=models.CharField(max_length=5,null=True,blank=True)
   sprice=models.IntegerField(blank=True,null=True)

   # def __str__(self):
   #    return self.size


class Category(models.Model):
   name=models.CharField(max_length=50,null=False,blank=False)

   def __str__(self):
      return self.name

class Product(models.Model):
   category=models.ForeignKey(Category,on_delete=models.SET_NULL ,null=True,blank=True)
   image=models.ImageField(null=False,blank=True)
   product_name=models.CharField(max_length=100,null=False)
   price=models.IntegerField()
   product_size=models.ForeignKey(Size,on_delete=models.SET_NULL,null=True,blank=True)


   def __str__(self):
      return self.product_name

class Order(models.Model):
   product=models.ForeignKey(Product,on_delete=models.CASCADE,blank=True)
   quantity=models.IntegerField(default=0)
   order_price=models.IntegerField(default=0)


class Profile(models.Model):
   user=models.OneToOneField(User,on_delete=models.CASCADE,null=True)
   image=models.ImageField(default='default.png',upload_to='user_pic')



class Fairtone(models.Model):
   category=models.ForeignKey(Category,on_delete=models.SET_NULL ,null=True,blank=True)
   image=models.ImageField(null=False,blank=True)
   product_name=models.CharField(max_length=100,null=False)
   price=models.IntegerField()
   product_size=models.ForeignKey(Size,on_delete=models.SET_NULL,null=True,blank=True)


   def __str__(self):
      return self.product_name
   
   
class Mediumtone(models.Model):
   category=models.ForeignKey(Category,on_delete=models.SET_NULL ,null=True,blank=True)
   image=models.ImageField(null=False,blank=True)
   product_name=models.CharField(max_length=100,null=False)
   price=models.IntegerField()
   product_size=models.ForeignKey(Size,on_delete=models.SET_NULL,null=True,blank=True)


   def __str__(self):
      return self.product_name
   
class Darktone(models.Model):
   category=models.ForeignKey(Category,on_delete=models.SET_NULL ,null=True,blank=True)
   image=models.ImageField(null=False,blank=True)
   product_name=models.CharField(max_length=100,null=False)
   price=models.IntegerField()
   product_size=models.ForeignKey(Size,on_delete=models.SET_NULL,null=True,blank=True)


   def __str__(self):
      return self.product_name