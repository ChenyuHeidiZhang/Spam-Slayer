from django.urls import path
from . import views

urlpatterns = [
	path('', views.home, name='slayer-home'),
	path('yelp/', views.yelp, name='slayer-yelp')
]
