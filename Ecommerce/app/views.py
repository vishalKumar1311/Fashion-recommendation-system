from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from .models import Product,Order,Category,Size,Profile,Fairtone,Mediumtone,Darktone
from .forms import Pimg
from django.core.paginator import Paginator
from django.contrib import messages
import os
import pickle
import tensorflow
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
# Create your views here.

def home(request):

   # img=os.listdir('./static/image/tshirt')
   category=request.GET.get('q')
   if category==None or category=='':
      products=Product.objects.all()
      paginator = Paginator(products, 8)
      page_number = request.GET.get('page')
      page_obj = paginator.get_page(page_number)
   else:
      products=Product.objects.filter(category__name=category)
      paginator = Paginator(products, 8)
      page_number = request.GET.get('page')
      page_obj = paginator.get_page(page_number)

   categorie=Category.objects.all()
  

   id=[1,2,3,4]
   pcount=[]
   for i in id:
      count=Product.objects.filter(category_id=i).count()
      pcount.append(count)
   
   categories=zip(categorie,pcount)
   # products=Product.objects.all()
   cart_count=Order.objects.all().count
   
   def step2(url):
         def extractSkin(image):
            # Taking a copy of the image
            img =  image.copy()
            # Converting from BGR Colours Space to HSV
            img =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
         
            #   Defining HSV Threadholds
            lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
            upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
      
            # Single Channel mask,denoting presence of colours in the about threshold
            skinMask = cv2.inRange(img,lower_threshold,upper_threshold)
      
            # Cleaning up mask using Gaussian Filter
            skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
      
            # Extracting skin from the threshold mask
            skin  =  cv2.bitwise_and(img,img,mask=skinMask)
      
            # Return the Skin image
            return cv2.cvtColor(skin,cv2.COLOR_HSV2BGR)
         def removeBlack(estimator_labels, estimator_cluster):
            # Check for black
            hasBlack = False
      
            # Get the total number of occurance for each color
            occurance_counter = Counter(estimator_labels)

      
            # Quick lambda function to compare to lists
            compare = lambda x, y: Counter(x) == Counter(y)
         
            # Loop through the most common occuring color
            for x in occurance_counter.most_common(len(estimator_cluster)):
         
               # Quick List comprehension to convert each of RBG Numbers to int
               color = [int(i) for i in estimator_cluster[x[0]].tolist() ]
         
      
         
               # Check if the color is [0,0,0] that if it is black 
               if compare(color , [0,0,0]) == True:
                  # delete the occurance
                  del occurance_counter[x[0]]
                  # remove the cluster 
                  hasBlack = True
                  estimator_cluster = np.delete(estimator_cluster,x[0],0)
                  break
            
         
            return (occurance_counter,estimator_cluster,hasBlack)
         def getColorInformation(estimator_labels, estimator_cluster,hasThresholding=False):
      
               # Variable to keep count of the occurance of each color predicted
               occurance_counter = None
      
               # Output list variable to return
               colorInformation = []
      
      
               #Check for Black
               hasBlack =False
      
               # If a mask has be applied, remove th black
               if hasThresholding == True:
         
                     (occurance,cluster,black) = removeBlack(estimator_labels,estimator_cluster)
                     occurance_counter =  occurance
                     estimator_cluster = cluster
                     hasBlack = black
         
               else:
                  occurance_counter = Counter(estimator_labels)
      
               # Get the total sum of all the predicted occurances
               totalOccurance = sum(occurance_counter.values()) 
      
      
               # Loop through all the predicted colors
               for x in occurance_counter.most_common(len(estimator_cluster)):
         
                  index = (int(x[0]))
         
                     # Quick fix for index out of bound when there is no threshold
                  index =  (index-1) if ((hasThresholding & hasBlack)& (int(index) !=0)) else index
         
                  # Get the color number into a list
                  color = estimator_cluster[index].tolist()
         
                  # Get the percentage of each color
                  color_percentage= (x[1]/totalOccurance)
         
                  #make the dictionay of the information
                  colorInfo = {"cluster_index":index , "color": color , "color_percentage" : color_percentage }
         
                  # Add the dictionary to the list
                  colorInformation.append(colorInfo)
         
            
               return colorInformation 
         def extractDominantColor(image,number_of_colors=5,hasThresholding=False):
      
               # Quick Fix Increase cluster counter to neglect the black(Read Article) 
               if hasThresholding == True:
                  number_of_colors +=1
      
               # Taking Copy of the image
               img = image.copy()
      
               # Convert Image into RGB Colours Space
               img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      
               # Reshape Image
               img = img.reshape((img.shape[0]*img.shape[1]) , 3)
      
               #Initiate KMeans Object
               estimator = KMeans(n_clusters=number_of_colors, random_state=0)
      
               # Fit the image
               estimator.fit(img)
      
               # Get Colour Information
               colorInformation = getColorInformation(estimator.labels_,estimator.cluster_centers_,hasThresholding)
               return colorInformation
         def plotColorBar(colorInformation):
            #Create a 500x100 black image
            color_bar = np.zeros((100,500,3), dtype="uint8")
      
            top_x = 0
            for x in colorInformation:    
               bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

            color = tuple(map(int,(x['color'])))
      
            cv2.rectangle(color_bar , (int(top_x),0) , (int(bottom_x),color_bar.shape[0]) ,color , -1)
            top_x = bottom_x
            return color_bar
         def prety_print_data(color_info):
            for x in color_info:
               print(pprint.pformat(x))
               print()  
         # Get Image from URL. If you want to upload an image file and use that comment the below code and replace with  image=cv2.imread("FILE_NAME")
         
         
         image =  imutils.url_to_image("http://127.0.0.1:8000"+url)

         # Resize image to a width of 250
         image = imutils.resize(image,width=250)

         #Show image
         # plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
         # plt.show()


         # Apply Skin Mask
         skin = extractSkin(image)

         # plt.imshow(cv2.cvtColor(skin,cv2.COLOR_BGR2RGB))
         # plt.show()



         # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors 
         dominantColors = extractDominantColor(skin,hasThresholding=True)

         color_value=[]
         for value in dominantColors:
            color_value.append(value['color'])

         print("Your Color is")

         if(color_value[0][0]>100 and color_value[0][1]>130
         and color_value[0][2]>100) and (color_value[1][0]>180 and color_value[1][1]>130
         and color_value[1][2]>100):
            skin_tone='Fair'
            print('Fair')
         elif(color_value[0][0]>160 and color_value[0][1]>80
         and color_value[0][2]>40) and (color_value[1][0]>160 and color_value[1][1]>80
         and color_value[1][2]>40):
            skin_tone='Medium'
            print('Medium')
         else:
            skin_tone="Dark"
            print("Dark") 
            
         return skin_tone         
   filename=Profile.objects.get(user=request.user)
   url=filename.image.url
   print(url)
   skin_tone=step2(url)  
   
   if skin_tone=="Fair":
      recommend_img=Fairtone.objects.all()
   elif skin_tone=="Medium":
      recommend_img=Mediumtone.objects.all()
   else:
      recommend_img=Darktone.objects.all()
   context={
      'recommend_img':recommend_img,
      'page_obj':page_obj,
      'cart_count':cart_count,
      'products':products,
      'categories':categories
   }
   return render(request,'app/index.html',context)


def tshirt(request):
   products=Product.objects.filter(category_id=1)
   paginator = Paginator(products, 8)
   page_number = request.GET.get('page')
   page_obj = paginator.get_page(page_number)
   categorie=Category.objects.all()
   id=[1,2,3]
   pcount=[]
   for i in id:
      count=Product.objects.filter(category_id=i).count()
      pcount.append(count)
   
   categories=zip(categorie,pcount)
   # products=Product.objects.all()
   cart_count=Order.objects.all().count
   context={
      'page_obj':page_obj,
      'cart_count':cart_count,
      'products':products,
      'categories':categories
   }
   return render(request,'app/index.html',context)

def shirt(request):
   products=Product.objects.filter(category_id=2)
   paginator = Paginator(products, 8)
   page_number = request.GET.get('page')
   page_obj = paginator.get_page(page_number)
   categorie=Category.objects.all()
   id=[1,2,3]
   pcount=[]
   for i in id:
      count=Product.objects.filter(category_id=i).count()
      pcount.append(count)
   
   categories=zip(categorie,pcount)
   # products=Product.objects.all()
   cart_count=Order.objects.all().count
   context={
      'page_obj':page_obj,
      'cart_count':cart_count,
      'products':products,
      'categories':categories
   }
   return render(request,'app/index.html',context)
def pants(request):
   products=Product.objects.filter(category_id=3)
   paginator = Paginator(products, 8)
   page_number = request.GET.get('page')
   page_obj = paginator.get_page(page_number)
   categorie=Category.objects.all()
   id=[1,2,3]
   pcount=[]
   for i in id:
      count=Product.objects.filter(category_id=i).count()
      pcount.append(count)
   
   categories=zip(categorie,pcount)
   # products=Product.objects.all()
   cart_count=Order.objects.all().count
   context={
      'page_obj':page_obj,
      'cart_count':cart_count,
      'products':products,
      'categories':categories
   }
   return render(request,'app/index.html',context)

def hoddies(request):
   products=Product.objects.filter(category_id=4)
   paginator = Paginator(products, 8)
   page_number = request.GET.get('page')
   page_obj = paginator.get_page(page_number)
   categorie=Category.objects.all()
   id=[1,2,3]
   pcount=[]
   for i in id:
      count=Product.objects.filter(category_id=i).count()
      pcount.append(count)
   
   categories=zip(categorie,pcount)
   # products=Product.objects.all()
   cart_count=Order.objects.all().count
   context={
      'page_obj':page_obj,
      'cart_count':cart_count,
      'products':products,
      'categories':categories
   }
   return render(request,'app/index.html',context)



def order(request):
   total=0
   show_order=Order.objects.all()
   price=request.GET.get('final_price')
   id=request.GET.get('id')
   order=Order()
   order.quantity=1
   if id is not None:
      product=Product.objects.get(pk=id)
      order.product=product
      order.order_price=price
      print("Something")
      order.save()
   for i in show_order:
      total+=i.order_price
   
   if request.method=='POST':
      add=request.POST.get('add')
      remove=request.POST.get('remove')
      obj=Order.objects.get(pk=add)
      price=Product.objects.get(id=obj.product_id)
      price=price.price
      if remove:
         obj.quantity-=1
         obj.order_price-=price
         obj.save()
      else:
         obj.quantity+=1
         obj.order_price+=price
         obj.save()

   context={
      'total':total,
      'show_order':show_order
   }
   return render(request,'app/cart.html',context)




def productDetails(request,id):
   all_size=Size.objects.all()
   cart_count=Order.objects.all().count
   curr_item=Product.objects.get(pk=id)
   similar=Product.objects.all().order_by('?')[:4]



  #ML PART

   feature_list = np.array(pickle.load(open('model/embeddings.pkl','rb')))
   filenames = pickle.load(open('model/filenames.pkl','rb'))

   model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
   model.trainable = False

   model = tensorflow.keras.Sequential([
      model,
      GlobalMaxPooling2D()
   ])
   sample_img=curr_item.image.url
   sample_img='static'+sample_img
   print(sample_img)
   img = image.load_img(sample_img,target_size=(224,224))
   img_array = image.img_to_array(img)
   expanded_img_array = np.expand_dims(img_array, axis=0)
   preprocessed_img = preprocess_input(expanded_img_array)
   result = model.predict(preprocessed_img).flatten()
   normalized_result = result / norm(result)

   neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
   neighbors.fit(feature_list)

   distances,indices = neighbors.kneighbors([normalized_result])

   # print(indices)
   
   image_list=[]
   for file in indices[0][1:6]:
      image_list.append(filenames[file])
      # print(filenames[file])

   final_list=[]
   for i in image_list:
      final_list.append(i[7:])
   
   # print(final_list)

   similarobj=[]
   for i in final_list:
      similar=Product.objects.get(image=i)
      similarobj.append(similar)
   
   print(similarobj)
   for i in similarobj:
      print(i.price)
   sobject=zip(final_list,similarobj)
   context={
      'sobject':sobject,
      'final_list':final_list,
      'cart_count':cart_count,
      'similar':similar,
      'curr_item':curr_item,
      'all_size':all_size
   }
   return render(request,'app/detail.html',context)



def remove(request,id):
   order=Order.objects.get(pk=id)
   order.delete()
   return redirect('ordered')





def cart(request):
   order_count=Order.objects.all().count()
   # for i in order:
   #    print(i.product.category)
   # print(order)
   context={
      'order_count':order_count
   }
   return render(request,'app/index.html',context)


def addToCart(request,id):
   product=Product.objects.get(pk=id)
   print(product)
   order=Order.objects.get(product=product)
   if order is not None:
      order.quantity=order.quantity+1
      order.save()
   else:
      order=order()
      order.quantity=order.quantity+1
      order.product=product
      order.save()
   return render(request,'app/index.html')



def signUp(request):
   profile=Pimg(instance=Profile)
   if request.method=='POST':
     
      username=request.POST.get('username')
      pass1=request.POST.get('pass1')
      pass2=request.POST.get('pass2')
      print(pass1)
      new_user=User.objects.create_user(username,'default@gmail.com',pass1)
      
      media=request.FILES or None
     
      profile=Pimg(request.POST,request.FILES)
      if profile.is_valid():
         obj=profile.save(commit=False)
         obj.user=new_user
         obj.save()
         return redirect('login')
   context={
      'profile':profile
   }
   return render(request,'app/signup.html',context)


def loginEW(request):
   if request.method=='POST':
      username=request.POST.get('username')
      password=request.POST.get('pass1')
      user=authenticate(username=username,password=password)
      print(user)
      if user is not None:
         print("Here")
         login(request, user)
         return redirect('home')
      else:
         messages.error(request, 'Invalid Credentials')

   return render(request,'app/login.html')