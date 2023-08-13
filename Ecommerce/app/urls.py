from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
urlpatterns = [
    path('', views.home,name='home'),
    path('detail/<int:id>',views.productDetails,name='detail'),
    path('order/<int:id>',views.addToCart,name='addToCart'),
    path('remove/<int:id>',views.remove,name='remove'),
    path('ordered/',views.order,name='ordered'),
    path('signup/',views.signUp,name='signup'),
    path('login/',views.loginEW,name='login'),
    path('Tshirt/',views.tshirt,name='Tshirt'),
    path('Shirt/',views.shirt,name='Shirt'),
    path('Pants/',views.pants,name='Pants'),
    path('Hoodies/',views.hoddies,name='Hoodies'),
]

urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
