import pennylane as qml
from pennylane import numpy as nnp
import torch
import numpy as np
import math as m
from tqdm import trange
from torch import nn
import torch.optim as optim
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import sklearn.datasets

from matplotlib.colors import ListedColormap

import os


def arqui(i,j):
    return (i==0 and j == 1) or (i==1 and j == 0) or (i==1 and j == 2) or (i==2 and j == 1) or (i==2 and j == 3) or (i==3 and j == 2) or (i==3 and j == 5) or (i==5 and j == 3) or (i==5 and j == 8) or (i==8 and j == 5) or (i==8 and j == 9) or (i==9 and j == 8) or (i==8 and j == 11) or (i==11 and j == 8) or (i==11 and j == 14) or (i==14 and j == 11) or (i==14 and j == 13) or (i==13 and j == 14) or (i==13 and j == 12) or (i==12 and j == 13) or (i==1 and j == 4) or (i==4 and j == 1) or (i==4 and j == 7) or (i==7 and j == 4) or (i==7 and j == 6) or (i==6 and j == 7) or (i==7 and j == 10) or (i==10 and j == 7) or (i==10 and j == 12) or (i==12 and j == 10) or (i==12 and j == 15) or (i==15 and j == 12) 


def Model_1(nq,nl):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(inputs,w):
        for i in range(nq):
            qml.RX( inputs[0],wires=i )
            qml.RZ( inputs[1],wires=i )
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq):
                for k in range(nq):
                    if j!=k:
                        #if arqui(j,k):
                        qml.CNOT(wires=[j,k])

        return qml.probs(wires=nq-1)    
    weight_shapes = {"w": (nq,nl) }
    qlayer = qml.qnn.TorchLayer(f, weight_shapes)
    return qlayer


def data_set(nTrain,nTest,Noise,dd):

    if dd == 1:
        x_train, y_train = sklearn.datasets.make_circles(nTrain,noise=Noise, factor=0.2, random_state=1)
        x_test, y_test =  sklearn.datasets.make_circles(nTest,noise=Noise, factor=0.2, random_state=1)
    else:
        x_train, y_train = sklearn.datasets.make_moons(nTrain,noise=Noise)
        x_test, y_test =  sklearn.datasets.make_moons(nTest,noise=Noise)


    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)

    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    return (x_train,y_train),(x_test,y_test)



def acc_val(xtest,ytest,model):
    acc = 0
    for i in range(xtest.shape[0]):
        out = model(xtest[i])[0]
        if out.item() >=0.5 and ytest[i].item() == 1:
            acc+=1
        if out.item() <0.5 and ytest[i].item() == 0:
            acc+=1
    return acc/xtest.shape[0]

def train_2( x_train, y_train,xtest,ytest,epochs,ii,nl,nq):
    loss = nn.MSELoss()
    loss_hist = []
    acc = []

    net =  Model_1(nq,nl)
        
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    tp = trange(epochs)
    for epoch in tp:
        soma_loss = 0

        for i in range(x_train.shape[0]):
            tp.set_description(f" nq: {nq}  nl: {nl} model: {ii+1} ")
            optimizer.zero_grad()
            out = net(x_train[i])[0]

            target = y_train[i].float()
            l = loss(out,target)
            l.backward()
            optimizer.step()
            soma_loss+=l.item()
        loss_hist.append(soma_loss/x_train.shape[0])
        acc.append( acc_val(xtest,ytest,net) )




    return np.array(loss_hist),np.array(acc)


    


(xtrain,ytrain),(xtest,ytest) = data_set(50,20,0.1,1)
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
plt.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain,  cmap=cm_bright, edgecolors="k")
plt.scatter(xtest[:, 0], xtest[:, 1], c=ytest,   edgecolors="k")
#plt.show()
plt.savefig('data.pdf',dpi=200)
plt.close()

epochs = 20
N = 50


for nq in [2,4,6,8,10]:
    name_model = 'model_nq_{}'.format(nq)
    if not os.path.exists('./{}'.format(name_model)):
        os.mkdir('./{}'.format(name_model))

    for nl in [2,4,6]:

        

        lr = 0.001
        hist = []
        acc = []
        for i in range(N):
            
            y,acc_=train_2(xtrain, ytrain,xtest,ytest,epochs,i,nl,nq)
            hist.append(y)
            acc.append(acc_)
            

        ## save
        y = np.array(hist)
        np.savetxt('./{}/loss_nq_{}_nl_{}_lr_{}.txt'.format(name_model,nq,nl,lr),y)

        z = np.array(acc)
        np.savetxt('./{}/acc_nq_{}_nl_{}_lr_{}.txt'.format(name_model,nq,nl,lr),z)


        grafico(hist,epochs,nq,nl,lr,name_model)
        grafico_acc(acc,epochs,nq,nl,lr,name_model)
