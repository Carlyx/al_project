from django.shortcuts import render
from django.shortcuts import redirect


# Create your views here.

def index(request):
    pass
    return render(request, 'index.html')


def contrast(request):
    pass
    return render(request, 'contrast.html')


def brightness(request):
    pass
    return render(request, 'brightness.html')


def filtering(request):
    pass
    return render(request, 'filtering.html')


def saturation(request):
    pass
    return render(request, 'saturation.html')
