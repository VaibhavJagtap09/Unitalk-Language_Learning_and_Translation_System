from django.urls import path, include 
from .import views 

urlpatterns = [

    path('', views.home ),
    path('home/', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('process_frame/', views.process_frame , name='process_frame'),    
]