'''
本程序意图获取扫描方式下的不同电压值
'''
from datetime import datetime
import pyqtgraph as pg
import array
import serial  # 导入串口包
import time  # 导入时间包
import numpy as np
import pandas as pd

import csv
import os
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout
import cv2

####################变量定义###################
k = 1
s = 0
length = 0
R_sensor0 = np.zeros((1, 16), dtype=np.float)#传感器迭代
R_sensor1 = np.zeros((1, 16), dtype=np.float)#传感器变化前的值
R_sensor2 = np.zeros((1, 16), dtype=np.float)
R_sensorN = np.zeros((1, 16), dtype=np.float)

rec_correct = list()
save_filename = r'D:\Python_project_2022\电压读取测试\adc8-0.584.csv'#文件保存地址与名字
####################串口检测###################
ser = serial.Serial('COM10', 115200, timeout=0.5)  # 开启com3口，波特率115200，超时0.5
ser.flushInput()  # 清空缓冲区
if ser.isOpen():
    print("open success!")
else:
    print("open error")
time.sleep(0.07)  # 延时0.07秒，免得CPU出问题
####################冗余数据裁剪函数###################
def data_cut(rec):
    ###################判断缓存数据完整性###################
    if rec.startswith("#") & rec.endswith("!"):
        # print("数据正常")
        pass
    else:
        # print("数据不正常")
        if rec[0] != '#':
            # print("数据头不对！，尝试切割")
            Del = rec.find("#")
            # print(Del)
            rec = rec[Del:]  # 去除读取到的不完整数据

        if rec[-1] != '!':
            # print("数据尾不对！，尝试切割")
            rec = rec[::-1]
            Del = rec.find("!")
            # print(Del)
            rec = rec[Del:]  # 去除读取到的不完整数据
            rec = rec[::-1]

        # print("数据修改结果：")
        # print(rec)
    # print("冗余数据切割完成")
    return rec
####################串口分析函数###################
def serial_analysis(rec):
    global length
    global s

    '''
    本函数用于删除缓存区的冗余数据并提取出一条后减少接收到的数据，需要搭配while函数使用
        案例：

    '''
    # print("数据分析器收到的数据：")
    # print(rec)

    ###################拆分数据###################
    start1 = rec.find("#")  # 找到第一个起始标志
    # print("第一个标志位为：", start1)
    end1 = rec.find("!")  # 找到第二个起始标志
    # print("第二标志位为：", end1)
    resistance = rec[start1 + 1:end1]  # 取出两个起始标志之间的字符串
    length = len(resistance)
    # print("string格式的电阻值为：", resistance)
    # print('\n')
    d_list = resistance.split(",")  # 以,为标志拆开 转换成一个list
    # print("list格式的电阻值为：", d_list)
    # print('\n')
    dint = list(map(float, d_list))  # str转换为float形数据
    # print(dint)
    # print('\n')
    d_arr = np.array(dint)  # 将list转换为array
    # print("数组格式的电阻值为：", d_arr)
    # print('\n')
    # print("缓存已提取%d次"%(s+1))

    return d_arr

####################数据保存函数###################
def data_save(Data,time,state):
    '''
    state=0:首次采集
    state=1：非首次采集
    '''
    if state==0:
        csv_head = ['Time(s)', '1.1-R(Kohm)', '1.2-R(Kohm)', '1.3-R(Kohm)', '1.4-R(Kohm)',
                    '2.1-R(Kohm)', '2.2-R(Kohm)', '2.3-R(Kohm)', '2.4-R(Kohm)',
                    '3.1-R(Kohm)', '3.2-R(Kohm)', '3.3-R(Kohm)', '3.4-R(Kohm)',
                    '4.1-R(Kohm)', '4.2-R(Kohm)', '4.3-R(Kohm)', '4.4-R(Kohm)']
        Data = np.insert(Data, 0, time, axis=1)  # 插入初始时间,第二个参数0表首位 axis=1表示按列插入
        # print(Data1.shape)
        Data = pd.DataFrame(Data)
        # print(Data1)
        Data.columns = csv_head
        Data.to_csv(save_filename, sep=',', float_format='%.3f', index=False, mode='w',
                    line_terminator='')  # 写数据，sep设置分隔符，index行索引，w写模型，line_terminator行之间不空行
    else:
            Data = np.insert(Data, 0, time, axis=1)  # 插入变化时间,第二个参数0表首位 axis=1表示按列插入
            # print(Data2.shape)
            Data = pd.DataFrame(Data)
            # print(Data2)
            Data.to_csv(save_filename, sep=',', float_format='%.3f', header=False, index=False,
                        mode='a', line_terminator='')  # 写数据，header列索引，a在源文件内容追加模型
    # print("数据已保存")

####################初始值读取与保存###################
# 读出数据作为初值

for m in range(2):#10
    time.sleep(0.1)  # 延时0.1秒，免得CPU出问题（10Hz采样）
    rec0 = ser.read(ser.in_waiting).decode("gbk")
    # print(rec0)
    rec0 = data_cut(rec0)#先把数据修剪一下
    index = rec0.count("#")
    while index != 0:
        R_sensor0 = serial_analysis(rec0)
        R_sensor1 = R_sensor1 + R_sensor0
        s+=1
        # print("************切除处理的数据START*************")
        rec0 = rec0[length+4:]#length只有数据，所以要多删除 # ！ / n 四个符号
        # print(rec0)
        index = rec0.count("#")
        # print(index)
        # print("************切除处理的数据END*************")
#计算初值均值
R_sensor1 = R_sensor1 / s  # 10次电阻均值
print("初始值是%d个元素的平均值"%(s))
Time0 = datetime.now()#捕获到初值后开始计时
data_save(R_sensor1,0.000,0)#保存数据
print("初始数据已保存")
######################绘图（曲线图与热力图）##############################
# 动态画图
app = pg.mkQApp()  # 建立app
win = pg.GraphicsLayoutWidget()  # 建立窗口
win.setWindowTitle('电阻阵列电阻曲线')
# win.resize(2, 500)  # 小窗口大小

win.setBackground('w')
# pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

data = array.array('d')  # 可动态改变数组的大小,double型数组
data1 = array.array('d')  # 可动态改变数组的大小,double型数组
data2 = array.array('d')  # 可动态改变数组的大小,double型数组
data3 = array.array('d')  # 可动态改变数组的大小,double型数组
data4 = array.array('d')  # 可动态改变数组的大小,double型数组
data5 = array.array('d')  # 可动态改变数组的大小,double型数组
data6 = array.array('d')  # 可动态改变数组的大小,double型数组
data7 = array.array('d')  # 可动态改变数组的大小,double型数组
data8 = array.array('d')  # 可动态改变数组的大小,double型数组
data9 = array.array('d')  # 可动态改变数组的大小,double型数组
data10 = array.array('d')  # 可动态改变数组的大小,double型数组
data11 = array.array('d')  # 可动态改变数组的大小,double型数组
data12 = array.array('d')  # 可动态改变数组的大小,double型数组
data13 = array.array('d')  # 可动态改变数组的大小,double型数组
data14 = array.array('d')  # 可动态改变数组的大小,double型数组
data15 = array.array('d')  # 可动态改变数组的大小,double型数组

historyLength = 500  # 横坐标长度

# 绘4*4子图
p = win.addPlot(row=0, col=0)  # 把图p加入到窗口中
p1 = win.addPlot(row=0, col=1)  # 把图p加入到窗口中
p2 = win.addPlot(row=0, col=2)  # 把图p加入到窗口中
p3 = win.addPlot(row=0, col=3)  # 把图p加入到窗口中
p4 = win.addPlot(row=1, col=0)  # 把图p加入到窗口中
p5 = win.addPlot(row=1, col=1)  # 把图p加入到窗口中
p6 = win.addPlot(row=1, col=2)  # 把图p加入到窗口中
p7 = win.addPlot(row=1, col=3)  # 把图p加入到窗口中
p8 = win.addPlot(row=2, col=0)  # 把图p加入到窗口中
p9 = win.addPlot(row=2, col=1)  # 把图p加入到窗口中
p10 = win.addPlot(row=2, col=2)  # 把图p加入到窗口中
p11 = win.addPlot(row=2, col=3)  # 把图p加入到窗口中
p12 = win.addPlot(row=3, col=0)  # 把图p加入到窗口中
p13 = win.addPlot(row=3, col=1)  # 把图p加入到窗口中
p14 = win.addPlot(row=3, col=2)  # 把图p加入到窗口中
p15 = win.addPlot(row=3, col=3)  # 0 把图p加入到窗口中
win.nextRow()
plot16 = win.addPlot()
item = pg.ImageItem()
plot16.addItem(item)

'''
# hist = pg.HistogramLUTItem()#加bar

# hist.setImageItem(item)


# win.addItem(hist)#加bar





# Item for displaying image data

# 设置坐标轴属性
# p.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p1.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p2.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p3.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p4.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p5.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p6.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p7.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p8.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p9.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p10.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p11.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p12.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p13.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p14.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)
# p15.setRange(xRange=[0, historyLength], yRange=[600, 2000], padding=0)

# 设置坐标轴属性
# p.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p1.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p2.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p3.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p4.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p5.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p6.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p7.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p8.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p9.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p10.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p11.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p12.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p13.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p14.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
# p15.setRange(xRange=[0, historyLength], yRange=[0, 1.2], padding=0)
'''

pen = pg.mkPen(color="b", width=2)# 颜色设置
curve = p.plot(pen=pen)
curve1 = p1.plot(pen=pen)
curve2 = p2.plot(pen=pen)
curve3 = p3.plot(pen=pen)
curve4 = p4.plot(pen=pen)
curve5 = p5.plot(pen=pen)
curve6 = p6.plot(pen=pen)
curve7 = p7.plot(pen=pen)
curve8 = p8.plot(pen=pen)
curve9 = p9.plot(pen=pen)
curve10 = p10.plot(pen=pen)
curve11 = p11.plot(pen=pen)
curve12 = p12.plot(pen=pen)
curve13 = p13.plot(pen=pen)
curve14 = p14.plot(pen=pen)
curve15 = p15.plot(pen=pen)

def plotData():
    global R_sensor2, R_sensorN
    Recv = ser.read(ser.in_waiting).decode("gbk")
    Recv = data_cut(Recv)#先把数据修剪一下
    index = Recv.count("#")
    # print("这次缓存区有%d个有效数据"%(index))
    while index != 0:
        R_sensor2 = serial_analysis(Recv)
        # print("************切除处理的数据START*************")
        Recv = Recv[length+4:]#length只有数据，所以要多删除 # ！ / n 四个符号
        # print(rec0)
        index = Recv.count("#")
        # print(index)
        # print("************切除处理的数据END*************")
        R_sensor2 = R_sensor2.reshape(1, 16)

        # print(R_sensorN)
        # print('\n')


        # 获取时间
        Time1 = datetime.now()
        D_time = Time1 - Time0
        T_time = D_time.microseconds / 1000000  # 毫秒转换为秒
        # T_time = D_time.seconds + round(T_time, 3)  # T_time需要作为x值输入
        T_time = D_time.seconds + float('%.3f' % T_time)
        # print("现在是第几秒：", T_time)

        data_save(R_sensor2,T_time,1)# 保存数据

        R_sensorN = R_sensor2
        # 数据推移
        if len(data) < historyLength:
            data.append(R_sensorN[0][0])
        else:
            data[:-1] = data[1:]  # 前移
            data[-1] = R_sensorN[0][0]
        curve.setData(data)

        if len(data1) < historyLength:
            data1.append(R_sensorN[0][1])
        else:
            data1[:-1] = data1[1:]  # 前移
            data1[-1] = R_sensorN[0][1]
        curve1.setData(data1)

        if len(data2) < historyLength:
            data2.append(R_sensorN[0][2])
        else:
            data2[:-1] = data2[1:]  # 前移
            data2[-1] = R_sensorN[0][2]
        curve2.setData(data2)

        if len(data3) < historyLength:
            data3.append(R_sensorN[0][3])
        else:
            data3[:-1] = data3[1:]  # 前移
            data3[-1] = R_sensorN[0][3]
        curve3.setData(data3)

        if len(data4) < historyLength:
            data4.append(R_sensorN[0][4])
        else:
            data4[:-1] = data4[1:]  # 前移
            data4[-1] = R_sensorN[0][4]
        curve4.setData(data4)

        if len(data5) < historyLength:
            data5.append(R_sensorN[0][5])
        else:
            data5[:-1] = data5[1:]  # 前移
            data5[-1] = R_sensorN[0][5]
        curve5.setData(data5)

        if len(data6) < historyLength:
            data6.append(R_sensorN[0][6])
        else:
            data6[:-1] = data6[1:]  # 前移
            data6[-1] = R_sensorN[0][6]
        curve6.setData(data6)

        if len(data7) < historyLength:
            data7.append(R_sensorN[0][7])
        else:
            data7[:-1] = data7[1:]  # 前移
            data7[-1] = R_sensorN[0][7]
        curve7.setData(data7)
        if len(data8) < historyLength:
            data8.append(R_sensorN[0][8])
        else:
            data8[:-1] = data8[1:]  # 前移
            data8[-1] = R_sensorN[0][8]
        curve8.setData(data8)

        if len(data9) < historyLength:
            data9.append(R_sensorN[0][9])
        else:
            data9[:-1] = data9[1:]  # 前移
            data9[-1] = R_sensorN[0][9]
        curve9.setData(data9)

        if len(data10) < historyLength:
            data10.append(R_sensorN[0][10])
        else:
            data10[:-1] = data10[1:]  # 前移
            data10[-1] = R_sensorN[0][10]
        curve10.setData(data10)

        if len(data11) < historyLength:
            data11.append(R_sensorN[0][11])
        else:
            data11[:-1] = data11[1:]  # 前移
            data11[-1] = R_sensorN[0][11]
        curve11.setData(data11)

        if len(data12) < historyLength:
            data12.append(R_sensorN[0][12])
        else:
            data12[:-1] = data12[1:]  # 前移
            data12[-1] = R_sensorN[0][12]
        curve12.setData(data12)

        if len(data13) < historyLength:
            data13.append(R_sensorN[0][13])
        else:
            data13[:-1] = data13[1:]  # 前移
            data13[-1] = R_sensorN[0][13]
        curve13.setData(data13)

        if len(data14) < historyLength:
            data14.append(R_sensorN[0][14])
        else:
            data14[:-1] = data14[1:]  # 前移
            data14[-1] = R_sensorN[0][14]
        curve14.setData(data14)

        if len(data15) < historyLength:
            data15.append(R_sensorN[0][15])
        else:
            data15[:-1] = data7[1:]  # 前移
            data15[-1] = R_sensorN[0][15]
        curve15.setData(data15)
    
        # R_sensorN = np.array([100,0,720,4,
        #                       2,2,2,550,
        #                       820,3,853,3,
        #                       4,4,4,1280], dtype=np.float)
        R_sensorN = 255-R_sensorN * 255 / R_sensorN.max() #转换成可识别的色阶255为最大
        img_color = cv2.cvtColor(R_sensorN.reshape(4, 4, 1).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # 灰度转彩色
        # print("##########color#############")
        # print(img_color)
        R_sensorN = cv2.applyColorMap(img_color, 4) #色彩模式
        item.setImage(cv2.transpose(cv2.flip(R_sensorN, flipCode=0))) #转置显示
        win.show()





timer = pg.QtCore.QTimer()
timer.timeout.connect(plotData)  # 定时调用plotData函数
timer.start(100)  # 多少ms调用一次
app.exec_()
