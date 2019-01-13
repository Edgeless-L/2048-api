#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras

from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import ExpectiMaxAgent,Agent2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
NUM_CLASSES = 4
NUM_EPOCHS = 20

display1 = Display()
display2 = IPythonDisplay()
model=keras.models.load_model('model.h5')

image=[]
label=[]
for i in range(0,10):
    game = Game(4, score_to_win=2048, random=False)
    agent = ExpectiMaxAgent(game, display=display1)
    
    while game.end==False:
        
        direction=agent.step()
        image.append(game.board)
        label.append(direction)
        game.move(direction)
       
    display1.display(game)
#运行10次游戏并记录棋盘和方向
x_train=np.array(image)
y_train=np.array(label)


x_train=np.log2(x_train+1)
x_train=np.trunc(x_train)
x_train = keras.utils.to_categorical(x_train, 12)

print(x_train.shape)
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

model.train_on_batch(x_train, y_train)       

# evaluate
score_train = model.evaluate(x_train, y_train, verbose=0)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0]*100,score_train[1]*100))

model.save('newmodel.h5')

image=[]
label=[]
count1=0
while count1<1:
    count=0
    for j in range(0,50):
        game = Game(4, score_to_win=2048, random=False)
        agent1 = ExpectiMaxAgent(game, display=display1)
        #agent2 = MyOwnAgent(game, display=display1)
        while game.end==False:
            direction1=agent1.step()#用强Agent判断方向
            
            x=np.array(game.board)
            
            x=np.log2(x+1)
            x=np.trunc(x)
            x = keras.utils.to_categorical(x, 12)
            x = x.reshape(1, 4, 4, 12)
            pred=model.predict(x,batch_size=128)
            r=pred[0]
            r1=r.tolist()
            direction2=r1.index(max(r1))
            #用新跑出来的模型判断方向
            image.append(game.board)
            label.append(direction1)#将用强Agent跑出来的方向作为训练的标签
            game.move(direction2)#用新模型判断的方向执行下一步
            
        display1.display(game)
        x=np.array(game.board)
        if np.amax(x)==2048:
            count+=1
        #如果本次游戏跑出2048，count+1
    if count>=48:
        break
    #如果50次游戏跑出48次2048，则循环结束
    x_train=np.array(image)
    y_train=np.array(label)
    x_train=np.log2(x_train+1)
    x_train=np.trunc(x_train)
    x_train = keras.utils.to_categorical(x_train, 12)
    #对棋盘进行12位one-hot编码


    print(x_train.shape)
   
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    #对方向标签进行4位one-hot编码
    
    model.train_on_batch(x_train, y_train)
    #用train_on_batch进行训练
    #evaluate
    score_train = model.evaluate(x_train, y_train, verbose=0)
    print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0]*100,score_train[1]*100))
    
    model.save('newmodel.h5')
    #记录模型并保存
    image=[]
    label=[]
    #清空棋盘和方向的矩阵
    print(i)
    print(count)
    
    
