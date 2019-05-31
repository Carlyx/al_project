from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from django.http import JsonResponse
from django.shortcuts import redirect
import json
import numpy as np
import time
from AI_project import numpy_sc3 as nsc
from AI_project import BP_net as hqh
import matplotlib.pyplot as plt
# Create your views here.
import os
import math



def index(request):
    pass
    return render(request, 'index.html')

def sortt(li):
    l1 = []
    l2 = []
    o = 0
    for i in li:
        o = max(len(i),o)
    for i in li:
        if len(i)<o:
            l1.append(i)
        else:
            l2.append(i)
    l1.sort()
    l2.sort()
    for i in l2:
        l1.append(i)
    return l1

def bp_pyx(request):
    if request.method == 'POST':
        hid_num = request.POST.get("in1")
        hid_lay = request.POST.get("in2")
        func = request.POST.get("func")
        act = request.POST.get("act")
        epoch = request.POST.get("in3")
        lr = request.POST.get("in4")
        print(hid_num, hid_lay, func, act, epoch, lr)
        for filename in os.listdir(r"E:/the third_2/AI/人工智能课设last/static/pyx_img/img/"):
            os.remove("E:/the third_2/AI/人工智能课设last/static/pyx_img/img/" + filename)
            # print('../static/img/'+filename)
        for filename in os.listdir(r"E:/the third_2/AI/人工智能课设last/static/pyx_img/img1/"):
            os.remove("E:/the third_2/AI/人工智能课设last/static/pyx_img/img1/" + filename)
        nsc.start(hid_num, hid_lay, func, act, epoch, lr)
        a = np.linspace(0, np.pi * 2, 100).reshape([100, 1])
        b = np.sin(a)
        c = np.cos(a)

        a = np.sum(a, axis=1)
        a = a.tolist()

        b = np.sum(b, axis=1)
        b = b.tolist()
        c = np.sum(c, axis=1)
        c = c.tolist()

    if request.is_ajax():
        if request.method == 'GET':
            print("jinru ")
            ll = []
            for filename in os.listdir(r"E:/the third_2/AI/人工智能课设last/static/pyx_img/img/"):
                ll.append('../static/pyx_img/img/' + filename)
            ll1 = []
            for filename in os.listdir(r"E:/the third_2/AI/人工智能课设last/static/pyx_img/img1/"):
                ll1.append('../static/pyx_img/img1/' + filename)
            ll = sortt(ll)
            ll1 = sortt(ll1)
            mm = {}
            mm1 = {}
            for ii in range(len(ll)):
                mm[ii] = ll[ii]
                mm1[ii] = ll1[ii]
            # return JsonResponse({"url": mm})
            return JsonResponse({"url": mm, "los": mm1})

    return render(request, "pyx.html")


def bp_hqh(request):
    if request.method == 'POST':
        hid_num = request.POST.get("in1")
        hid_lay = request.POST.get("in2")
        func = request.POST.get("func")
        act = request.POST.get("act")
        epoch = request.POST.get("in3")
        lr = request.POST.get("in4")
        print(hid_num, hid_lay, func, act, epoch, lr)
        # for filename in os.listdir(r"/Users/huqinhan/Desktop/人工智能应用/人工智能课设last/static/hqh_img/img/"):
        #     os.remove("/Users/huqinhan/Desktop/人工智能应用/人工智能课设last/static/hqh_img/img/" + filename)
        for filename in os.listdir(r"E:/the third_2/AI/人工智能课设last/static/hqh_img/img/"):
            os.remove("E:/the third_2/AI/人工智能课设last/static/hqh_img/img/" + filename)
            # print('../static/img/'+filename)
        hqh.start(hid_num, hid_lay, func, 1, epoch, lr)

    if request.is_ajax():
        if request.method == 'GET':
            print("jinru ")
            ll = []
            # for filename in os.listdir(r"/Users/huqinhan/Desktop/人工智能应用/人工智能课设last/static/hqh_img/img/"):
            for filename in os.listdir(r"E:/the third_2/AI/人工智能课设last/static/hqh_img/img/"):
                ll.append('../static/hqh_img/img/' + filename)

            ll = sortt(ll)
            print(ll)
            mm = {}

            for ii in range(len(ll)):
                mm[ii] = ll[ii]

            # return JsonResponse({"url": mm})
            return JsonResponse({"url": mm})

    return render(request, "hqh.html")



pointnum=0
def hyd(request):
    knowledge = list()
    num = list()
    tp = list()
    indu = dict()
    point = list()
    pointnum = 0

    def getknowledge(path):  # 得到知识库
        fr = open(path)
        for line in fr:
            x = list()
            number = 0
            str = ""
            for i in line:
                if i == " ":
                    x.append(str)
                    number = number + 1
                    str = ""
                elif i == "\n":
                    break
                else:
                    str = str + i
            x.append(str)
            number = number + 1
            knowledge.append(x)
            num.append(number)

    path = "E:/the third_2/AI/人工智能课设last/AI_project/知识库.txt"
    getknowledge(path)
    return render(request,'hyd.html',{"knowledge":knowledge})


def exper(request):
    knowledge = list()
    num = list()
    tp = list()
    indu = dict()
    point = list()
    global pointnum
    def getknowledge(path):  # 得到知识库
        fr = open(path)
        for line in fr:
            x = list()
            number = 0
            str = ""
            for i in line:
                if i == " ":
                    x.append(str)
                    number = number + 1
                    str = ""
                elif i == "\n":
                    break
                else:
                    str = str + i
            if str=="":
                continue
            x.append(str)
            number = number + 1
            knowledge.append(x)
            num.append(number)

    def getpoint():  # 得到点
        cnt = 0
        global pointnum
        for l in knowledge:
            for i in l:
                flag = False
                for j in point:
                    if i == j:
                        flag = True
                        break
                if flag == False:
                    point.append(i)
                    cnt += 1
                    pointnum += 1
        # print(point)

    def getindu():  # 得到入度
        cnt = 0
        for l in knowledge:
            for i in l:
                indu[i] = 0
        for l in knowledge:
            for i in range(num[cnt] - 1, num[cnt]):
                str = l[i]
                indu[str] += num[cnt] - 1
            cnt += 1

    def tuopu():  # 拓扑排序
        for pi in range(0, pointnum):
            vis = ""
            for pj in point:
                if indu[pj] == 0:
                    vis = pj
                    # print(vis)
                    tp.append(pj)
                    indu[pj] -= 1
                    break
            if vis == "":
                continue
            else:
                cnt = 0
                for l in knowledge:
                    flag = False
                    for i in range(0, num[cnt] - 1):
                        if l[i] == vis:
                            flag = True
                            break
                    if flag == True:
                        for i in range(num[cnt] - 1, num[cnt]):
                            # print(l[i])
                            if indu[l[i]] > 0:
                                indu[l[i]] -= 1
                            # print(indu[l[i]])
                    cnt += 1
        # print(tp)

    def judge(str):  # 判断正反向推理
        cnt = 0
        for l in knowledge:
            for i in range(0, num[cnt] - 1):
                if l[i] == str:
                    return True
            cnt += 1
        return False

    def gettext(path):
        text = list()
        fr = open(path)
        for line in fr:
            str = ""
            for i in line:
                if i == "\n":
                    break
                str += i
            if str=="":
                continue
            text.append(str)
        # print(text)
        return text

    def Forward_reasoning(path):  # 正向推理
        ans=list()
        text = gettext(path)
        flag = True
        while (flag):
            flag = False
            cnt = 0
            for l in knowledge:
                f = True
                for i in range(0, num[cnt] - 1):
                    if l[i] not in text:
                        f = False
                        break
                if f == True:
                    for i in range(0, num[cnt] - 1):
                        text.remove(l[i])
                    for i in range(num[cnt] - 1, num[cnt]):
                        text.append(l[i])
                    flag = True
                    x = list()
                    for i in range(0, num[cnt] - 1):
                        x.append(l[i])
                    print(x, end="")
                    print("->", end="")
                    print(l[num[cnt] - 1])
                    ss=""
                    for k in x:
                        ss += k+" "
                    ss+="->"
                    ss+=l[num[cnt] - 1]
                    ans.append(ss)
                cnt += 1
        print(text)
        if len(text) > 1:
            ans=list()
            ans.append("条件不足，无法得到正确推理!")
            print("条件不足，无法得到正确推理!")
        return ans

    def Backward_reasoning(path):  # 反向推理
        text = gettext(path)
        ans=list()
        str = text[0]
        flag = True
        while (flag):
            flag = False
            cnt = 0
            for l in knowledge:
                for i in range(num[cnt] - 1, num[cnt]):
                    if l[i] in text:
                        flag = True
                        tot = 0
                        x = list()
                        for k in knowledge:
                            for j in range(num[tot] - 1, num[tot]):
                                if k[j] == l[i]:
                                    for kk in range(0, num[tot] - 1):
                                        if k[kk] not in text:
                                            text.append(k[kk])
                                            x.append(k[kk])
                            tot += 1
                        print(l[i], end="")
                        print("->", end="")
                        print(x)
                        ss=""
                        ss+=l[i]
                        ss+="->"
                        for mm in x:
                            ss+=mm+" "
                        ans.append(ss)
                        text.remove(l[i])
                cnt += 1
        if str == text[0]:
            print("知识库中不存在,无法推理!")
            return
        print(str, end="")
        print("->", end="")
        print(text)
        ss=""
        ss+=str
        ss+="->"
        for mm in text:
            ss+=mm+" "
        ans.append(ss)
        return ans
    path="E:/the third_2/AI/人工智能课设last/AI_project/知识库.txt"
    getknowledge(path)
    # getpoint()
    # getindu()
    # tuopu()
    path="/Users/huqinhan/Desktop/人工智能应用/人工智能课设last/AI_project/推理.txt"
    text = gettext(path)
    if request.method == 'POST':
        if request.POST.get('Forward')=='Forward':
            print("2343")
            forward=request.POST.get('input')
            print(forward)
            f=open("E:/the third_2/AI/人工智能课设last/AI_project/推理.txt",'w')
            f.write(forward)
            f.close()
            ans= Forward_reasoning(path)
            return render(request, 'forward.html', {"ans":ans})
        elif request.POST.get('Backward')=='Backward':
            backward=request.POST.get('input')
            f = open("E:/the third_2/AI/人工智能课设last/AI_project/推理.txt", 'w')
            f.write(backward)
            f.close()
            ans=Backward_reasoning(path)
            return render(request, 'forward.html', {"ans": ans})
        elif request.POST.get('change')=='change':
            backward=request.POST.get('input')
            print(backward)
            f = open("E:/the third_2/AI/人工智能课设last/AI_project/知识库.txt", 'w')
            f.write(backward)
            f.close()
            return HttpResponse("修改知识库成功")
    return HttpResponse("123")