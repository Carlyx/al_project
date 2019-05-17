from django.shortcuts import render
from django.shortcuts import redirect


# Create your views here.

def index(request):
    pass
    return render(request, 'index.html')


def bp_pyx(request):
    pass
    return render(request,'pyx.html')