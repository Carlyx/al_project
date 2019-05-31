import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

class Floor(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)
        self.input_data = None
        self.output_data = None

    def get_input(self):
        return self.input_dim

    def get_output(self):
        return self.output_dim

    def store_input(self, input_data):
        self.input_data = input_data

    def store_output(self, output_data):
        self.output_data = output_data

    def check(self):
        print(self.w, self.b)

# (拟合的函数1.sin(x) 2.cos(x) 3.x^2 4.x^2+y^2)
def init(func):
    global x_train, y_train,z_train, x_test, y_test, z_test

    if func == 1:
        x_train = np.linspace(0, 2 * np.pi, 100).reshape([100, 1])
        y_train = np.sin(x_train)

        x_test = np.array(np.random.rand(100, 1) * np.pi * 2)
        y_test = np.sin(x_test)
        return x_train, y_train, x_test, y_test
    elif func == 2:
        x_train = np.linspace(-3, 3, 100).reshape([100, 1])

        y_train = x_train ** 2 + x_train ** 3

        x_test = []
        for i in range(100):
            x_test.append([random.random() * 6 - 3])
        # print(x_test)
        x_test = np.array(x_test)

        y_test = x_test ** 2 + x_test ** 3

        return x_train, y_train, x_test, y_test
    elif func == 4:
        x_train = np.linspace(-3, 3, 100).reshape([100, 1])

        y_train = x_train * x_train

        x_test = []
        for i in range(100):
            x_test.append([random.random() * 6 - 3])
        # print(x_test)
        x_test = np.array(x_test)

        y_test = x_test * x_test

        return x_train, y_train, x_test, y_test
    elif func == 3:
        x_train = np.linspace(0, 2 * np.pi, 100).reshape([100, 1])
        y_train = np.sin(x_train) + np.cos(x_train)

        x_test = np.array(np.random.rand(100, 1) * np.pi * 2)
        y_test = np.sin(x_test) + np.cos(x_test)
        return x_train, y_train , x_test, y_test
        # x_train = np.linspace(-3, 3, 100).reshape([100, 1])
        # y_train = np.linspace(-3, 3, 100).reshape([100, 1])
        # z_train = x_train**2 + y_train**2
        #
        # x_test = np.array(np.random.rand(100,1)*3)
        # y_test = np.array(np.random.rand(100,1)*3)
        # z_test = x_test**2 + y_test**2
        # return x_train, y_train, z_train, x_test, y_test, z_test
    else:
        x_train = np.linspace(-3, 3, 100).reshape([100, 1])
        y_train = x_train**3 + x_train**2 + x_train

        x_test = []
        for i in range(100):
            x_test.append([random.random() * 6 - 3])
        # print(x_test)
        x_test = np.array(x_test)
        x_test = np.array(np.random.rand(100, 1) * 3)
        y_test = x_test**3 + x_test**2 + x_test
        return x_train, y_train, x_test, y_test
    # plt.plot(x_data,y_data)
    # plt.show()
    # plt.plot(x,y)
    # plt.show()


def matrix_train(a):
    res = []
    for i in range(a.shape[1]):
        res.append([0 for i in range(a.shape[0])])

    for i in range(a.shape[1]):
        for j in range(a.shape[0]):
            res[i][j] = a[j][i]

    return res

def matrix_mul(a,b):
    res = []
    for i in range(a.shape[0]):
        res.append([0 for j in range(b.shape[1])])
    co = 0

    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                res[i][j] += a[i][k] * b[k][j]

    return res

def matrix_add(a,b):
    res = []
    for i in range(a.shape[0]):
        res.append([0 for i in range(a.shape[1])])
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            res[i][j] = a[i][j] + b[i][j]
    return res

def matrix_dot_mul(a,b):
    res = []
    for i in range(a.shape[0]):
        res.append([0 for j in range(a.shape[1])])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            res[i][j] = a[i][j]*b[i][j]
    return res

class Sigmoid(object):
    def __init__(self, dim):
        self.input_dim = dim
        self.output_dim = dim
        self.res = []
        self.input_data = None
        self.output_data = None

    def activate(self, x):
        self.input_data = x
        self.res = []
        self.output_data = (1 + np.exp(-x)) ** -1
        # print(self.output_data)
        return self.output_data
        # self.res = 1/(1+np.exp(-x))

    def get_input(self):
        return self.input_dim

    def get_output(self):
        return self.output_dim

    def get_res(self):
        return self.res

    def back(self, gradient):
        # print(gradient)
        #   每次为 后一层的 W*gradient
        return gradient * (self.output_data - self.output_data ** 2)

class Relu(Floor):


    def __init__(self, dim):
        self.input_dim = dim
        self.output_dim = dim
        self.input_data = None
        self.output_data = None

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = input_data*(input_data>0)
        return self.output_data

    def back(self, gradient, lr=0.01):
        return gradient * np.ones(self.output_data.shape)*(self.output_data > 0)

class Net(object):
    def __init__(self, x_test, y_test, gradient_func=None, lr=0.5, ls_func=None):
        self.x_test = x_test
        self.y_test = y_test
        self.floor = []
        self.activate = []
        self.gradient_func = gradient_func
        self.lr = lr
        self.ls_func = ls_func

    def add_floor(self, floor):
        if len(self.floor) == 0:
            self.floor.append(floor)
        elif self.floor[-1].get_output() != floor.get_input():
            exit("层数不一致")
        else:
            self.floor.append(floor)

    def add_activate(self, activate):
        self.activate.append(activate)

    def check(self):
        for i in self.floor:
            print(i.get_input(), i.get_output())

    # x 为每一个batch_size大小的数据，i为第i层
    def forward(self, x, i):
        # 初始为 (10,1)*(1*3) 3为第一层神经元个数，1为有影响的输入个数，乘后为（10,3）每一行为一个独立的运算过程，即每一行为一个输入的运算，一共10行表示一次性处理10个输入
        self.floor[i].store_input(x)
        data = x
        # print(data.shape,self.floor[i].weights.shape)
        output = np.matmul(data, self.floor[i].weights) + self.floor[i].bias
        self.floor[i].store_output(output)
        # print(output.shape)
        if i < len(self.activate):
            output = self.activate[i].activate(output)
        # print("after",output.shape)
        return output

    # 返回前向传播的结果
    def predict(self, x):
        _x = x
        for i in range(len(self.floor)):
            _x = self.forward(_x, i)
        return _x

    # 实际输出-理想输出， lr， 第i层网络
    def back(self, gradient, lr, i):
        d_weights = np.zeros(self.floor[i].weights.shape)
        d_biass = np.zeros(self.floor[i].bias.shape)
        gradient_news = []
        for inp, gra in zip(self.floor[i].input_data, gradient):
            d_weight, d_bias, gradient_new = self.__back(inp, gra, lr, i)
            d_weights += d_weight
            d_biass += d_bias
            gradient_news.append(gradient_new)
        # 有多个输入，so求平均值
        self.floor[i].weights -= d_weights / len(gradient)
        self.floor[i].bias -= d_biass / len(gradient)
        return np.array(gradient_news)
        # print(d_weights.shape,d_bias.shapenn)

    def __back(self, input_data, gradient, lr, i):
        # neto1 对w求导为outh1 即输入部分，所以用输入的来乘ls再乘lr
        '''
            [
                [a1 a1 a1]                       [
                [a2 a2 a2]     *  gradient  =       [ w11 w12 w13]
                [a3 a3 a3]                          [ w21 w22 w23]
            ]                                       [ w31 w32 w33] ]
        '''
        d_weight = np.array([input_data for i in gradient]).T * gradient * lr
        d_bias = gradient * lr
        # 因为一次性考虑多个输入，求总和
        # print("=====-------------------")
        # print(self.floor[i].weights.shape)
        # print(np.sum(self.floor[i].weights,axis=1).shape)
        # print("===============")
        # print("SAdas==============================")
        # print(self.floor[i].weights.shape)
        # print(np.sum(self.floor[i].weights,axis=1).shape)
        gradient_new = np.sum(self.floor[i].weights, axis=1) * gradient  # 记录用来求前一层的gradient
        return d_weight, d_bias, gradient_new

    def train(self, x_data, y_data, batch_size=100, epochs=1000, callable=None):
        for i in range(epochs):
            if callable:
                callable(i)
            for j in range(0, len(x_data), batch_size):
                batch_x = x_data[j:j + batch_size]
                batch_y = y_data[j:j + batch_size]
                y_predict = self.predict(batch_x)

                # 反向传播
                # 传入实际输出以及正确输出, 即总ls对第一个输出的求导（targeto1-outo1）
                gradient = self.gradient_func(y_predict, batch_y)
                for k in range(len(self.floor) - 1, -1, -1):
                    if k < len(self.activate):
                        gradient = self.activate[k].back(gradient)
                        # 此时把总ls对第一个输出激活前的求导 返回
                    gradient = self.back(gradient, self.lr, k)




ls1 = []
ls2 = []
ss = []

coo = 0
coolos = 0
def speed1(i):
    global coo,coolos
    global net, x_train, y_train, x_test, y_test, ss
    if i % 100 == 0:
        los1 = net.ls_func(net.predict(x_train), y_train)
        los2 = net.ls_func(net.predict(x_test), y_test)
        print('epochs: %d ls1=%f, ls2=%f acc=%f' % (i, los1 / 100, los2 / 100, np.exp(-los1 / 100)))
        ls1.append(los1)
        ls2.append(los2)
    if i % 500 == 0:
        # plt.figure(1)
        # plt.subplot(1,2,1)
        plt.plot(range(len(ls1)), ls1, 'b-')
        # plt.plot(range(len(ls2)), ls2, 'r-')
        plt.savefig("E:/the third_2/AI/人工智能课设last/static/pyx_img/img1/test" + str(coolos) + ".jpg")
        coolos+=1
        ls1.clear()
        ls2.clear()
        plt.clf()
        # plt.show()
        y_predict = net.predict(x_test)
        x_predict = net.predict(x_train)
        sss = []
        sss.append(x_train.tolist())
        sss.append(x_predict.tolist())
        ss.append(sss)

        # plt.subplot(1,2,2)
        plt.plot(x_train.tolist(), x_predict.tolist(), 'g+')
        plt.plot(x_test.tolist(), y_predict.tolist(), 'r+')
        plt.plot(x_train.tolist(), y_train.tolist(), 'b-')
        plt.savefig("E:/the third_2/AI/人工智能课设last/static/pyx_img/img/test"+str(coo)+".jpg")
        coo+=1
        # print(coo)

        plt.clf()
        # plt.show()

def speed2(i):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    global net, x_train, y_train, x_test, y_test, z_train, z_test
    if i % 100 == 0:
        los1 = net.ls_func(net.predict(np.hstack([x_train,y_train])), z_train)
        los2 = net.ls_func(net.predict(np.hstack([x_test, y_test])), z_test)
        print('epochs: %d ls1=%f, ls2=%f acc=%f' % (i, los1 / 100, los2 / 100, np.exp(-los1 / 100)))
        ls1.append(los1)
        ls2.append(los2)
    if i % 1000 == 0:
        y_predict = net.predict(np.hstack([x_train,y_train]))
        ax.plot(x_train, y_train, y_predict, label='parametric curve')
        ax.legend()

        plt.show()


def main(lrr, inp_numm, hide_numm, hide_layerr, funcc, actt, epochs):
    # print(x_train[0:3])
    # print(y_train[0:3])
    # exit()
    global net
    # if funcc == 3:
    if funcc == 10:
        net = Net(np.hstack([x_train, y_train]), z_train, gradient_func=lambda y_, y: y_ - y, lr=lrr,
                  ls_func=lambda y_, y: np.sum(0.5 * (y_ - y) ** 2)/20)
        net.add_floor(Floor(inp_numm, hide_numm))
        for i in range(hide_layerr):
            net.add_activate(Sigmoid(hide_numm))
            net.add_floor(Floor(hide_numm, hide_numm))
        net.add_activate(Sigmoid(hide_numm))
        net.add_floor(Floor(hide_numm, inp_numm))
        # net.add_activate(Sigmoid(1))
        net.train(np.hstack([x_train, y_train]), z_train, batch_size=20, epochs=epochs, callable=speed2)
    else:
        if actt == 1:
            net = Net(x_train, y_train, gradient_func=lambda y_, y: y_ - y, lr=lrr,
                      ls_func=lambda y_, y: np.sum(0.5 * (y_ - y) ** 2)/20)
            net.add_floor(Floor(inp_numm, hide_numm))
            for i in range(hide_layerr):
                net.add_activate(Sigmoid(hide_numm))
                net.add_floor(Floor(hide_numm, hide_numm))
            net.add_activate(Sigmoid(hide_numm))
            net.add_floor(Floor(hide_numm, inp_numm))
            # net.add_activate(Sigmoid(1))
            net.train(x_train, y_train, batch_size=20, epochs=epochs, callable=speed1)
        else:
            net = Net(x_train, y_train, gradient_func=lambda y_, y: y_ - y, lr=lrr,
                      ls_func=lambda y_, y:np.sum(0.5*(y_-y)**2))
            net.add_floor(Floor(inp_numm, hide_numm))
            for i in range(hide_layerr):
                net.add_activate(Sigmoid(hide_numm))
                net.add_floor(Floor(hide_numm, hide_numm))
            net.add_activate(Sigmoid(hide_numm))
            net.add_floor(Floor(hide_numm, inp_numm))
            # net.add_activate(Sigmoid(1))
            net.train(x_train, y_train, batch_size=20, epochs=epochs, callable=speed1)
    # net.train(x_test, y_test, batch_size=20, epochs=7100, callable=speed1)
    # net.check()

# 前端数据
# 隐藏层神经元个数 隐藏层层数 (拟合的函数1.sin(x) 2.cos(x) 3.sin(x)+cos(y) 4.x^2 5 x5+x4+x3+x2+x) 激活函数(1.sigmoid 2.relu)  lr
def get_data_from_the_front_end(inp_num=1, hide_num=19, hide_layer=2, func=1, act=1, lr=0.01, epoch=5100):
    global net, x_train, y_train, z_train, x_test, y_test, z_test,coo,coolos
    coo=0
    coolos=0
    if func == 3:
        x_train, y_train, x_test, y_test = init(func)
        # x_train, y_train, z_train, x_test, y_test, z_test = init(func)
    else:
        x_train, y_train, x_test, y_test = init(func)
    main(lr, inp_num, hide_num, hide_layer, func, act, epoch)


def start(hid_num, hid_lay, func, act, epoch, lr):
    hid_num = int(hid_num)
    hid_lay = int(hid_lay)
    print(type(hid_lay))
    if func == '3':
        # get_data_from_the_front_end(2, int(hid_num), int(hid_lay), 3, int(act), float(lr), int(epoch))
        get_data_from_the_front_end(1, int(hid_num), int(hid_lay), int(func), int(act), float(lr), int(epoch))
    else:
        get_data_from_the_front_end(1, int(hid_num), int(hid_lay), int(func), int(act), float(lr), int(epoch))

count = 0
def get_data():
    global count
    print("count",count)
    if len(ss)==0:
        pass
    elif count >= len(ss)-1:
        count+=1
        return ss[-1]
    else:
        count+=1
        return ss[count-1]
# if __name__ == '__main__':
#    main()


'''
     sin函数最优:
     get_data_from_the_front_end(1,15,2,1,1,1,0.05)
     get_data_from_the_front_end(1,16,2,1,1,2,0.03)
     
     cos
     get_data_from_the_front_end(1,16,2,1,2,1,0.03)
     
     x^2
     15 2 2100 0.03 
     
     x^3 + x^2 
     16 2 2600 0.01 x^3 + x^2   sigmoid
     
     sinx + cosx
     16 2 2100 0.03 xinx+cosx sigmoid
'''
