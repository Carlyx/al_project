import numpy as np
import matplotlib.pyplot as plt
import time
import random
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z):
    """
    激活函数sigmoid
    :param z:
    :return:
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_de(z):
    """
    求sigmoid函数的导数
    :param z:
    :return:
    """
    return sigmoid(z) * (1 - sigmoid(z))


def forward(X, W, B):
    """
    :param X: 1*m
    :param W: 权重列表，包含各层之间的权重 w1:1*3 w2:3*1
    :param b:
    :return: z1 a1:3*m  z2 a2:1*m A:[a1,a2]
    """
    A = []
    Z = []
    z0 = W[0].T.dot(X) + B[0]
    Z.append(z0)
    a0 = sigmoid(z0)
    A.append(a0)

    count = 0
    index = 1
    while count < len(W) - 1:
        z = W[index].T.dot(A[index - 1]) + B[index]
        a = sigmoid(z)
        Z.append(z)
        A.append(a)
        index += 1
        count += 1

    return Z, A


def cost_single(target, net_out):
    """
    对一个训练数据计算代价函数
    :param target:
    :param net_out:
    :return: Ei:1*m
    """
    loss = target - net_out
    Ei = 1 / 2 * (loss * loss)
    # 输出层神经元数量大于1
    if target.shape[0] > 1:
        # axis=0压缩行 axis=1压缩列
        Ei = np.sum(Ei, axis=0)
    return Ei


def cost(X, Y, A):
    """
    所有训练数据的总体（平均）代价
    :param X: 1*m
    :param Y: 1*m 目标输出
    :param A: [a1,a2] a2:实际输出 1*m
    :return: E_total: float
    """
    # 样本数量
    N = X.shape[1]
    # A列表的最后一个元素是前向传播的输出结果，即a2
    Ei = cost_single(Y, A[len(A) - 1])
    # 压缩列，因为列数代表样本个数，即把所有单个样本的代价加在一起
    E_total = Ei.sum(axis=1) * 1 / N
    # E_total: List[]
    return float(E_total)


def backward(Layer, X, Y, Z, A, W):
    # 假设 1--3--2
    # delta_List
    delta_list = []

    # 输出层delta : 2*m
    delta_L = -(Y - A[len(A) - 1]) * sigmoid_de(Z[len(Z) - 1])
    delta_list.append(delta_L)

    # #所有单个样本的E对W的梯度的和 : 3*2 和W[len(W)-1]一致
    # E_w_total = A[len(A)-2].dot(delta_L.T)

    # 从第L-1层到第2层依次计算隐藏层的delta,更新参数
    # 4.25 怎么解决角标问题 4.26 用设置多个标记的方法解决
    # 设置各个List的标记
    index_w = len(W) - 1
    index_delta = 0
    index_z = len(Z) - 2

    for i in range(len(Layer) - 2):
        delta_l = (W[index_w].dot(delta_list[index_delta])) * sigmoid_de(Z[index_z])
        delta_list.append(delta_l)
        # 标记更新
        index_w -= 1
        index_delta += 1
        index_z -= 1

    # 存储每一层 所有样本的代价函数E对参数w的偏导数的List
    E_w_total_list = []
    # 代价函数对参数w、b的偏导数
    index_a = len(A) - 1
    for i in range(len(delta_list)):
        if i < len(delta_list) - 1:
            # 所有(N)样本在l层的E对W_l的偏导数的和 : 3*2 和W[len(W)-1]一致
            E_wl_total = A[index_a - 1].dot(delta_list[i].T)
            E_w_total_list.append(E_wl_total)
            index_a -= 1
        # 遍历到了输入层
        elif i == len(delta_list) - 1:
            E_wl_total = X.dot(delta_list[i].T)
            E_w_total_list.append(E_wl_total)

    # #转换为重输入层到输出层
    # E_b_list = list(reversed(delta_list))
    # E_w_total_list = list(reversed(E_w_total_list))
    E_b_list = delta_list
    return E_w_total_list, E_b_list


def update(W, B, E_w_total_list, E_b_list, X, lr):
    N = X.shape[1]
    index = 0
    for i in range(len(W) - 1, -1, -1):
        W[i] = W[i] - (lr / N) * E_w_total_list[index]
        # E_b_total = float(np.sum(E_b_list[index],axis=1))
        E_b_total = np.sum(E_b_list[index], axis=1).reshape(E_b_list[index].shape[0], 1)
        B[i] = B[i] - (lr / N) * E_b_total

        index += 1


def parameter_init(Layer):
    W = []
    B = []
    for i in range(len(Layer) - 1):
        wi = np.random.randn(Layer[i], Layer[i + 1])
        bi = np.random.randn(Layer[i + 1], 1)
        W.append(wi)
        B.append(bi)

    return W, B


def init(func):
    global x_train, y_train, z_train, x_test, y_test, z_test
    if func == 1:
        x_train = np.linspace(0, 2 * np.pi, 150, dtype=float).reshape(1, 150)
        y_train = (np.sin(x_train) + 1) / 2

        # x_train = x_data[:, 0:int(0.8 * x_data.shape[1])]
        # y_train = y_data[:, 0:int(0.8 * y_data.shape[1])]

        # x_test = x_data[:, 0:(x_data.shape[1] - x_train.shape[1])]
        # y_test = x_data[:, 0:(y_data.shape[1] - y_train.shape[1])]
        x_test = np.random.sample((1, 150)) * 2 * np.pi
        y_test = (np.sin(x_test) + 1) / 2

        return x_train, y_train, x_test, y_test

    elif func == 4:
        x_train = np.linspace(-1, 1, 100, dtype=float).reshape(1, 100)
        y_train = x_train * x_train



        x_test = np.linspace(-1, 1, 100, dtype=float).reshape(1, 100)
        y_test = x_test * x_test
        return x_train, y_train, x_test, y_test

    elif func == 3:
        x_train = np.linspace(-1, 1, 100, dtype=float).reshape(1, 100)
        y_train = np.linspace(0, 2, 100, dtype=float).reshape(1, 100)
        z_train = (np.sin(x_train) + np.cos(y_train))

        x_test = x_train
        y_test = y_train
        z_test = z_train

        return x_train, y_train, z_train, x_test, y_test, z_test


# 前端数据
# 隐藏层神经元个数 隐藏层层数 (拟合的函数1.sin(x) 2.cos(x) 3.sin(x)+cos(y) 4.x^2 5 x5+x4+x3+x2+x) 激活函数(1.sigmoid 2.relu)  lr
def get_data_from_the_front_end(inp_num=1, hide_num=19, hide_layer=2, func=1, act=1, lr=0.01, epoch=5100):
    global net, x_train, y_train, z_train, x_test, y_test, z_test, coo, coolos
    coo = 0
    coolos = 0
    if func == 3:
        x_train, y_train, z_train, x_test, y_test, z_test = init(func)
        main(lr, inp_num, hide_num, hide_layer, func, act, epoch, x_train, y_train, x_test, y_test, z_train, z_test)
    else:
        x_train, y_train, x_test, y_test = init(func)
        main(lr, inp_num, hide_num, hide_layer, func, act, epoch, x_train, y_train, x_test, y_test)


def start(hid_num, hid_lay, func, act, epoch, lr):
    hid_num = int(hid_num)
    hid_lay = int(hid_lay)
    print(type(hid_lay))
    if func == '3':
        get_data_from_the_front_end(2, int(hid_num), int(hid_lay), 3, int(act), float(lr), int(epoch))
    else:
        get_data_from_the_front_end(1, int(hid_num), int(hid_lay), int(func), int(act), float(lr), int(epoch))


def main(lr, inp_num, hide_num, hide_layer, func, act, epoch, x_train, y_train, x_test, y_test, z_train=0,
         z_test=0):
    global coolos
    start_time = time.time()
    # X = np.linspace(0, 2 * np.pi, 150, dtype=float).reshape(1, 150)
    # # noise = np.random.normal(0, 0.05, X.shape)
    # Y = (np.sin(X) + 1) / 2
    #
    # x_data = np.linspace(0, 2 * np.pi, 150, dtype=float).reshape(1, 150)
    # y_data = (np.sin(x_data) + 1) / 2
    #
    # x_train = x_data[:, 0:int(0.8 * x_data.shape[1])]
    # y_train = y_data[:, 0:int(0.8 * y_data.shape[1])]
    #
    # # x_test = x_data[:, 0:(x_data.shape[1] - x_train.shape[1])]
    # # y_test = x_data[:, 0:(y_data.shape[1] - y_train.shape[1])]
    # x_test = np.random.sample((1, 150)) * 2 * np.pi
    # y_test = (np.sin(x_test) + 1) / 2

    Layer = [inp_num]
    for i in range(hide_layer):
        Layer.append(hide_num)
    Layer.append(1)
    # Layer = [1, 80, 100, 80, 1]

    W, B = parameter_init(Layer)
    # lr = 0.1
    step = 0
    y_loss = []

    if func != 3:
        while step < epoch:
            Z, A = forward(x_train, W, B)
            E_w_total_list, E_b_list = backward(Layer, x_train, y_train, Z, A, W)
            update(W, B, E_w_total_list, E_b_list, x_train, lr)
            # if step%100 == 0:
            #     print("cost:" + str(cost(X, Y, A)))
            x_loss = np.arange(0, step + 1, 1)

            y_loss.append(cost(x_train, y_train, A))

            if step % 1000 == 0:
                print("epochs:%d" % step)
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1)
                # plt.subplot(1, 2, 1)
                ax1.plot(x_train, y_train, 'g+')
                ax1.plot(x_train, A[len(A) - 1], 'r.')

                ax2 = fig.add_subplot(1, 2, 2)
                ax2.plot(x_loss, y_loss, 'b.')

                # plt.show()
                # fig.savefig("/Users/huqinhan/Desktop/人工智能应用/人工智能课设last/static/hqh_img/img/test" + str(coolos) + ".jpg")
                fig.savefig("E:/the third_2/AI/人工智能课设last/static/hqh_img/img/test" + str(coolos) + ".jpg")
                coolos += 1
            step += 1
        end_time = time.time()
        print("Training time = %s" % (end_time - start_time))

        Z_test, A_test = forward(x_test, W, B)

        plt.figure()
        plt.plot(x_test, y_test, 'g+')
        plt.plot(x_test, A_test[len(A_test) - 1], 'r.')
        # plt.savefig("/Users/huqinhan/Desktop/人工智能应用/人工智能课设last/static/hqh_img/img/test" + str(coolos) + ".jpg")
        plt.savefig("E:/the third_2/AI/人工智能课设last/static/hqh_img/img/test" + str(coolos) + ".jpg")
        # plt.show()

    else:
        xy_train = np.concatenate((x_train, y_train), axis=0)
        xy_test = xy_train
        while step < epoch:
            Z, A = forward(xy_train, W, B)
            E_w_total_list, E_b_list = backward(Layer, xy_train, z_train, Z, A, W)
            update(W, B, E_w_total_list, E_b_list, xy_train, lr)
            # if step%100 == 0:
            #     print("cost:" + str(cost(X, Y, A)))
            x_loss = np.arange(0, step + 1, 1)

            y_loss.append(cost(x_train, z_train, A))

            if step % 1000 == 0:
                print("epochs:%d" % step)

                # plt.subplot(1, 2, 1)
                X, Y = np.meshgrid(x_train, y_train)

                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')

                surf = ax1.plot_surface(X, Y, z_train, rstride=1, cstride=1,
                                       linewidth=0, antialiased=False)
                surf2 = ax1.plot_surface(X, Y, A[len(A)-1], rstride=1, cstride=1,
                                         linewidth=0, antialiased=False)


                ax2 = fig.add_subplot(1, 2, 2)
                ax2.plot(x_loss, y_loss, 'b.')

               # plt.show()
                fig.savefig("/Users/huqinhan/Desktop/人工智能应用/人工智能课设last/static/hqh_img/img/test" + str(coolos) + ".jpg")
                coolos += 1
            step += 1
        end_time = time.time()
        print("Training time = %s" % (end_time - start_time))

        Z_test, A_test = forward(xy_test, W, B)

        # plt.figure()
        # plt.plot(x_test, y_test, 'g+')
        # plt.plot(x_test, A_test[len(A_test) - 1], 'r.')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 2, 1, projection='3d')
        # plt.subplot(1, 2, 1)
        X, Y = np.meshgrid(x_train, y_train)
        surf3 = ax2.plot_surface(X, Y, z_train, rstride=1, cstride=1,
                                linewidth=0, antialiased=False)

        surf2 = ax2.plot_surface(X, Y, z_test, rstride=1, cstride=1,
                                linewidth=0, antialiased=False)

        #ax2.set_zlim3d(-1, 1)
        fig2.savefig("/Users/huqinhan/Desktop/人工智能应用/人工智能课设last/static/hqh_img/img/test" + str(coolos) + ".jpg")
        # plt.show()
