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



class param:
    def __init__(self, n,l=1):
        self.n = n
        self.l = l
        self.y = 0
        self.Novo = None
        self.inputs = []
        self.w = []
    def create_circuit(self,gate):
       
        self.dev = qml.device('default.qubit',wires=self.n)
        @qml.qnode(self.dev, interface="torch")
        def net(inputs,w):
            
            for i in range(self.n):
                qml.RX( inputs[0],wires=i )
                qml.RZ( inputs[1],wires=i )

            for k in range(self.l):
               
                for i in range(self.n):
                    if gate[0][i] == 0:
                        qml.RX(w[i][k] ,wires=i)
                    if gate[0][i] == 1:
                        qml.RY( w[i][k],wires=i)
                    if gate[0][i] == 2:
                        qml.RZ( w[i][k],wires=i)
                    
                ss = 0
                for i in range(self.n):
                    for j in range(self.n):
                        if i!=j:
                            bb = gate[0][ss+self.n]
                            
                            if bb == 1:
                                qml.CNOT(wires=[i,j])
                            if bb == 2:
                                qml.CY(wires=[i,j])
                            if bb == 3:
                                qml.CZ(wires=[i,j])
                            
                            
                            ss+=1
                    

            return qml.expval(qml.PauliZ(wires=self.n-1)) 
        return net
    
    
    def torchLayer(self):
        """
        funcao que ira integrar o circuito PENNYLANE em PYTORCH 
        """
        
        weight_shapes = {"w": (self.n,self.l) }
        qlayer = qml.qnn.TorchLayer(self.Novo, weight_shapes)
        
        return qlayer

    def reset(self):
        gate = np.random.choice(3,(1,self.n))
        conec = np.random.choice(4,(1,self.n*(self.n-1)))
        self.y = np.concatenate((gate, conec), axis=1)
        #print(self.y)
        self.Novo = self.create_circuit(self.y)
        #return self.novo
    
    def render(self):
        if len(self.inputs) == 0: 
            self.inputs = np.random.random(2)
        if len(self.w) == 0:
            self.w = np.random.random((self.n,self.l))
        fig, ax = qml.draw_mpl(self.Novo, decimals=4)(self.inputs,self.w)
        plt.show()
        

        
    def novo(self):
        gate = np.random.choice(3,(1,self.n))
        conec = np.random.choice(4,(1,self.n*(self.n-1)))
        x = np.concatenate((gate, conec), axis=1)
       
        self.Novo = self.create_circuit(x)
        



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
        out = model(xtest[i])
        if out.item() >=0.5 and ytest[i].item() == 1:
            acc+=1
        if out.item() <0.5 and ytest[i].item() == 0:
            acc+=1
    return acc/xtest.shape[0]

def train_2(model, x_train, y_train,xtest,ytest,epochs,ii,nl,nq):
    loss = nn.MSELoss()
    loss_hist = []
    acc = []

    net =  model.torchLayer()
        
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    tp = trange(epochs)
    for epoch in tp:
        soma_loss = 0

        for i in range(x_train.shape[0]):
            tp.set_description(f" nq: {nq}  nl: {nl} model: {ii+1} ")
            optimizer.zero_grad()
            out = net(x_train[i])

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
N = 300


for nq in [2,4,6,8,10]:
    name_model = 'model_nq_{}'.format(nq)
    if not os.path.exists('./{}'.format(name_model)):
        os.mkdir('./{}'.format(name_model))

    for nl in [2,4,6]:

        model11 =  param(nq,nl)
        model11.reset()

        lr = 0.001
        hist = []
        acc = []
        for i in range(N):
            
            y,acc_=train_2(model11, xtrain, ytrain,xtest,ytest,epochs,i,nl,nq)
            hist.append(y)
            acc.append(acc_)
            model11.novo()

        ## save
        y = np.array(hist)
        np.savetxt('./{}/loss_nq_{}_nl_{}_lr_{}.txt'.format(name_model,nq,nl,lr),y)

        z = np.array(acc)
        np.savetxt('./{}/acc_nq_{}_nl_{}_lr_{}.txt'.format(name_model,nq,nl,lr),z)


        