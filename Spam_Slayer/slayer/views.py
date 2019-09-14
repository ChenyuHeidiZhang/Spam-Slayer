from django.shortcuts import render
#from .models import reviews

reviews = [
    {
        'author': 'CoreyMS',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'August 27, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'August 28, 2018'
    }
]

def home(request):
    if (request.method == 'POST'): 
        url = request.POST.get('url-input', None)

    context = {
        'reviews': reviews
    }
    return render(request, 'slayer/home.html', context)

def yelp(request):
    context = {
        'reviews': reviews
    }
    return render(request, 'slayer/yelp.html', context)
