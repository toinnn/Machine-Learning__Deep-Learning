import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import pickle
import heapq
import pandas as pd


def oneHotEncode(dim, idx , device = "cpu"):
    vector = torch.zeros(dim , device = device)
    vector[idx] = 1
    return vector

class oneHotClass(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self ,dim , idx , device ):
        return oneHotEncode(dim , idx , device = device)

# 2 versões : A = 1 backward p/ kda contexto , B = 1 backward p/ todos os contextos ao msm tempo
class skip_gram_MLP(nn.Module):
    def __init__(self, topology, inputSize , device = torch.device("cpu")):
        super().__init__()
        
        self.layer = [nn.Linear(inputSize, topology[0])]
        for i in range(1, len(topology)):
            self.layer.append(nn.Linear(topology[i - 1], topology[i]))
        
        
        if torch.device("cpu") == device :
            self.dim = torch.tensor([inputSize])
            self.layer = nn.Sequential(*self.layer)
            
        else :
            self.dim = torch.tensor([inputSize]).to(device = device)
            self.layer = nn.Sequential(*self.layer).to(device = device)
            

        
        

    def fit(self, n, maxAge, maxErro ,momentum = 0.5 , inputBatch = None, targetBatch = None ,loadBatch = None, lossFunction = nn.NLLLoss() ,device = torch.device("cpu") ):
        x_plt = []
        y_plt = []
        bestModel = self.layer
        minErro = np.inf
        # inputBatch = (torch.tensor([i]).cuda() for i in inputBatch )
        #### inputBatch = torch.tensor(inputBatch )
        # targetBatch= (torch.tensor([i]).cuda() for i in targetBatch )
        #### targetBatch = torch.tensor(targetBatch )
        # n = torch.tensor([n]).cuda()
        # if torch.cuda.is_available :
        print(torch.device("cuda") , device )
        if torch.device("cuda") == device :
            oneHot = oneHotClass.cuda(self)
            # oneHot = oneHot
            maxAge = torch.tensor([maxAge]).cuda()
            maxErro = torch.tensor([maxErro]).cuda()
            momentum = torch.tensor([momentum]).cuda()
            optimizer = torch.optim.SGD(self.parameters(), lr=n, momentum = momentum)
            loss = None
            # lossFunction = nn.BCEWithLogitsLoss()
            self.cuda()
            for i in range(maxAge):
                print("Época de treino : {}".format(i))
                aux = 0
                auxAge = 0
                lastModel = self.layer
                ctd = 0
                for (x, y) in loadBatch :

                    # if torch.cuda.is_available :
                    # if torch.device("cuda") == device :
                    out = self.forward(oneHotEncode(self.dim ,x , device = "cuda" ))
                    ctd += 1
                    if ctd % 500 == 0 :
                        print(ctd)
                    # loss = F.nll_loss( self.y[-1].view(1,-1) , y)
                    # print(y)
                    # print("y[-1] : {}".format( self.y[-1].view(1,-1).shape)) 
                    if loss == None :
                        loss = lossFunction( out.view(1,-1).cuda() , y.cuda() )
                    else :
                        loss += lossFunction( out.view(1,-1) , y )
                    # loss = lossFunction(y, self.y[-1])
                    # print(loss.item())

                    aux += loss.item()
                    auxAge += 1

                loss = loss/len(loadBatch)
                loss = loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                
                
                y_plt.append(aux / auxAge)
                print("erro médio : {}".format(y_plt[-1]))

                if y_plt[-1] < minErro:
                    minErro = y_plt[-1]
                    bestModel = lastModel
                if y_plt[-1] <= maxErro:
                    break
        else:
            
            optimizer = torch.optim.SGD(self.parameters(), lr=n, momentum = momentum)
            for i in range(maxAge):
                print("Época de treino : {}".format(i))
                aux = 0
                auxAge = 0
                lastModel = self.layer
                ctd = 0
                for x, y in zip(inputBatch, targetBatch):
                    
                    out = self.forward(oneHotEncode(self.dim,x ))
                    ctd += 1
                    if ctd % 500 == 0 :
                        print(ctd)
                    # loss = F.nll_loss( self.y[-1].view(1,-1) , y)
                    # print(y)
                    # print("y[-1] : {}".format( self.y[-1].view(1,-1).shape))
                    loss = lossFunction( out.view(1,-1) , y )
                    # loss = lossFunction(y, self.y[-1])
                    # print(loss.item())

                    aux += loss.item()
                    auxAge += 1

                    loss = loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()
                # for j in self.layer:
                    # optimizer = torch.optim.SGD(j.parameters(), lr=n, momentum = momentum)
                    # optimizer = torch.optim.Adadelta(j.parameters(),lr = n)

                    # optimizer.step()
                    # optimizer.zero_grad()

                y_plt.append(aux / auxAge)
                print("erro médio : {}".format(y_plt[-1]))

                if y_plt[-1] < minErro:
                    minErro = y_plt[-1]
                    bestModel = lastModel
                if y_plt[-1] <= maxErro:
                    break
        x_plt = list(range(len(y_plt)))
        plt.plot(x_plt, y_plt)
        plt.show()
        self.layer = bestModel

    def forward(self, x):
        # self.y = [self.layer[0](x)]
        # for i in range(1, len(self.layer)):
        #     self.y += [F.log_softmax(self.layer[i](self.y[-1])  ,dim = 0  )]# , dim = 0

        return self.layer(x)
class dataset(torch.utils.data.Dataset) :
    def __init__(self,data , transform = False ,dim = None , device = "cpu"):
        self.data = data
        self.transfor = transform
        self.dim = dim
        self.device = device
        self.id2key = list(data.keys())
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        if self.transfor :
            idx = self.id2key[idx]
            # print(self.data)
            # print(type(idx))
            return (oneHotEncode(self.dim , idx , device = self.device) ,
                torch.tensor(self.data[idx].to(device = torch.device(self.device)) ).float())
        else :
            key = self.id2key[idx]
            return (key , self.data[key] )
    # def __getstate__(self):

    #     return __dict__
    
    # def __setstate__(self,state):
    #     self.__dict__ = state

# class dataLoader(torch.utils.data.DataLoader) :


class skip_gram():
    def __init__(self,corpus = None):
        # super(skip_gram,self).__init__()
        self.corpus = corpus
        self.tokens = None #{Tokens : indice associado}
        self.mlp = None
        self.vocabulary = None #{Token : torch.tensor() associado}
        self.tknCorpus = None#corpus mas com as palavras trocadas pelos indices dos tokens
        self.idx2token = []#list(pos_associada->token)
        # self.vec2idx = {}#{torch.tensor() : indice associado}

    def tokenize(self,corpus=None):
        if corpus == None :
            if self.corpus == None :
                print("nenhum corpus oferecido")
            else :
                self.tokens = [i.split() for i in self.corpus]
        else :
            self.corpus = corpus
            self.tokens = [i.split() for i in corpus]

        dicio = {}
        for i in self.tokens:
            for j in i:
                dicio[j] = None
        dicio = {j: i for i, j in zip(range(len(dicio.keys())), dicio.keys())}
        self.tknCorpus = [[dicio[word] for word in seq] for seq in self.tokens]
        self.tokens = dicio
        return self.tknCorpus
    
    def sequence2idx(self,sequence ):
        tkn = sequence.split()
        idx = []
        for tk in tkn :
            idx += [self.tokens[tk]]
        return idx

    def pairsList(self,corpus = None , window =2) -> list :
        pairs = []
        if corpus != None :
            self.tokenize(corpus)
        for seq in self.tknCorpus :
            for centerWordPos in range(len(seq)) :
                for contextPos in range(centerWordPos - window ,centerWordPos + window  +1) :
                    if contextPos <0 or contextPos >= len(seq) or contextPos == centerWordPos :
                        continue
                    pairs.append([seq[centerWordPos],seq[contextPos]])
        return pairs

    def trainVectors(self,vectorDim ,n,momentum, maxAge , maxErro ,pairsSaved=None,saveNewPairs = False , window = 5 , corpus = None ,
        device = torch.device("cpu")):
        print("Treinamento de vetores iniciado")
        if self.tokens == None :
            print("O corpus prescisa primeiro ser tokenizado")
            return
        if pairsSaved == None :
            pairs = self.pairsList(corpus = corpus ,window = window )
            input, target = zip(*pairs)
        else :
            input, target = pairsSaved
        print("Passou do pairs")

        if saveNewPairs :
            with open("savedPairs.json","w+") as saveFile :
                json.dump([input , target] , saveFile)
            print("Novo pairs json salvo com sucesso")
        # input , target = [self.oneHotEncode(i) for i in input],[torch.tensor([t]) for t in target]
        target = [torch.tensor([t]) for t in target]

        print(len(input))
        if torch.device("cpu") != device :
            loader = DataLoader(dataset(dict(pairs),transform = False ,dim = len(self.tokens) , device = "cuda") , batch_size = 1 ,
                num_workers = 7 ,shuffle=True)
            self.mlp = skip_gram_MLP((vectorDim , len(self.tokens)),len(self.tokens)).cuda()
            self.mlp.fit(n , maxAge , maxErro ,loadBatch = loader , momentum = momentum , device = torch.device("cuda") )
        else :
            print("veio de cpu")
            self.mlp = skip_gram_MLP((vectorDim , len(self.tokens)),len(self.tokens))
            self.mlp.fit(n , maxAge , maxErro , inputBatch = input , targetBatch = target , momentum = momentum  )
        columns = [torch.tensor(i) for i in list(zip(*self.mlp.layer[0].weight))]
        self.vocabulary = {word:column for column , word in  zip( columns , self.tokens.keys() ) }
        # print(self.vocabulary)
        self.idx2token = [ i for i in self.vocabulary.keys() ]
        
        
        return self.vocabulary
    def __setitem__(self,token , vector) :
        if token not in self.tokens.keys() :
            lengh = len(self.idx2token)
            self.vocabulary[token] = vector
            self.tokens[token] = lengh
            self.idx2token.append(token)
        else :
            print("Token já faz parte do vocabulário ")
            return None

    def sequence2vectors(self , sequence):
        tkn = []
        for i in sequence.split() :
            tkn += [self.vocabulary[i].view(1,-1)]
        return torch.cat( tuple(tkn) , dim = 0)
    def __len__(self):
        return len(self.vocabulary)
