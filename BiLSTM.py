import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import random as rd
from matplotlib import pyplot as plt
import copy as cp
from torch.nn.utils.rnn import pad_sequence
from overloading.overloading import override , overload
from pyspark.sql import SparkSession

print("teste")
def diff_Rate(a,b):
    smallerSize = min(len(a) , len(b))
    correct = 0
    for i in range(smallerSize):
        print("a[i] {} -- b[i] {}".format(a[i] , b[i]))
        # for j in 
        if a[i]==b[i] :
            correct += 1 
    biggerSize = max(len(a) , len(b))
    return 1 - correct/biggerSize 

class Encoder(nn.Module):
    def __init__(self , input_dim , hidden_size , num_Layers , device = torch.device("cpu") ):
        super(Encoder , self).__init__()
        self.hidden_size = hidden_size
        self.input_dim   = input_dim
        self.num_Layers  = num_Layers
        self.lstm        = nn.LSTM(input_dim , hidden_size , num_Layers , batch_first = True , bidirectional = True).to(device)
        # self.linear      = nn.Linear(hidden_size*2 , num_classes )
        self.device      = device

    def setDevice(self , device):
        self.device = device
        self.lstm   = self.lstm.to(device)
        # self.linear.to(device)
    
    def forward(self , x , hidden , cell ) :
        # h0 = torch.zeros(self.num_Layers*2 , len(x) , self.hidden_size )
        # c0 = torch.zeros(self.num_Layers*2 , len(x) , self.hidden_size )
        
        x , (hidden , cell )  = self.lstm(x , (hidden , cell))

        # return self.lstm(x , (h0 , c0))[1]
        return x , (hidden , cell)

class Decoder(nn.Module):
    def __init__(self , input_dim , hidden_size , num_Layers , num_classes , device = torch.device("cpu") ):
        super(Decoder , self).__init__()
        self.hidden_size = hidden_size
        self.input_dim   = input_dim
        self.num_Layers  = num_Layers
        self.lstm        = nn.LSTM(input_dim , hidden_size , num_Layers , batch_first = True , bidirectional = True).to(device)
        self.linear      = nn.Linear(hidden_size*2 , num_classes ).to(device)
        self.device      = device

    def setDevice(self , device):
        self.device = device
        self.lstm   = self.lstm.to(device)
        self.linear = self.linear.to(device)
    
    def forward(self , x , hidden_State , cell_State) :
        # h0 = torch.zeros(self.num_Layers*2 , x.size(0) , self.hidden_size )
        # c0 = torch.zeros(self.num_Layers*2 , x.size(0) , self.hidden_size )
        
        x , (hidden_State , cell_State) = self.lstm(x , (hidden_State , cell_State))

        return self.linear(x[ : , -1 , : ]) , (hidden_State , cell_State)
 
class BiLSTM(nn.Module ):#, override):
    def __init__(self , input_dim , hidden_size_Encoder , num_Layers_Encoder ,
            hidden_size_Decoder , num_Layers_Decoder , num_classes , embedding , EOS_Vector , device = torch.device("cpu") ):
        super(BiLSTM , self).__init__()
        
        self.input_dim   = input_dim
        self.device      = device
        
        self.hidden_size_Encoder = hidden_size_Encoder
        self.num_Layers_Encoder  = num_Layers_Encoder
        self.hidden_size_Decoder = hidden_size_Decoder 
        self.num_Layers_Decoder  = num_Layers_Decoder
        self.num_classes         = num_classes

        self.encoder   = Encoder( input_dim , hidden_size_Encoder , num_Layers_Encoder , device)
        self.decoder   = Decoder(  input_dim , hidden_size_Decoder , num_Layers_Decoder , num_classes , device )
        # self.attention = nn.Linear(2*hidden_size_Encoder*num_Layers_Encoder + 2*hidden_size_Decoder*num_Layers_Decoder , 1 )
        #    hidden_size_Decoder*num_Layers_Decoder*2 )
        self.embedding = embedding
        self.EOS = EOS_Vector
        self.BOS = -EOS_Vector
        #type(.stoi) == type({"token":index}) , type(.itos) == type(["token",..]) , type(vectors)==type([vec0,vec1,vec2,..]) 

    def setDevice(self , device):
        self.device = device
        self.encoder.setDevice(device)
        self.decoder.setDevice(device)
        self.EOS = self.EOS.to(self.device) 
        self.BOS = self.BOS.to(self.device)

    def forward(self , x , out_max_Len = 150 , master_Imput = None , encoder_State = None ) :
        #ENCODER :
        
        if encoder_State == None :
            hidden_State = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
            cell_State   = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )

            _ , (hidden , cell) = self.encoder(x.view(1 , x.shape[0] , x.shape[1] ).to(self.device) , hidden_State , cell_State )

        else :
            hidden , cell = encoder_State



        #DECODER :
        seq = []
        out_class_Seq = []
        if master_Imput == None :
            
            buffer = self.BOS.view(1,1,-1).to(self.device)
            ctd = 0
            states = []
            while (buffer  != self.EOS.to(self.device)).all() and len(seq) < out_max_Len :
                
                out , (hidden , cell) = self.decoder(buffer.view(1,1,-1) , hidden , cell) 
                # print(out[0])
                states += [(hidden , cell)]
                out        = heapq.nlargest( 1 , enumerate( out[0] ) , key = lambda x : x[1] )[0]

                
                word   = self.embedding.itos[ out[0] ]
                buffer = self.embedding[ word ].float().to(self.device)

                seq           += [word] 
                out_class_Seq += [out[0]]
                ctd   += 1
                
            return seq , torch.tensor(out_class_Seq).to(self.device) , states
            # return self.linear(x[ : , -1 , : ])
        else :
            
            out , (hidden , cell) = self.decoder(master_Imput , hidden , cell)
            out        = heapq.nlargest( 1 , enumerate( out[0] ) , key = lambda x : x[1] )[0]

            word   = self.embedding.itos[ out[0] ]
            buffer = self.embedding[ word ].float().to(self.device)

            seq           += [word] 
            out_class_Seq += [out[0]]

            return seq , torch.tensor(out_class_Seq).to(self.device) , (hidden , cell)


    def forward_fit(self , x , out_max_Len = 150 ,target = None , force_target_input_rate = 0.5 , master_Imput = None ,
        encoder_State = None) :
        
        # x = x.view(1 , x.shape[0] , x.shape[1] )
        #ENCODER :
        if encoder_State == None :
            hidden_State = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
            cell_State   = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
            
            _ , (hidden , cell) = self.encoder(x.view(1 , x.shape[0] , x.shape[1] ).to(self.device) , hidden_State , cell_State )
        else :
            hidden , cell = encoder_State 
        
        #DECODER :
        out_seq = []
        if master_Imput == None :
            buffer = self.BOS.view(1,1,-1).to(self.device)
            states = []
            ctd = 0
            # teste = (buffer  != self.EOS.to(self.device)).all()
            # print("hidden.shape ",hidden.shape)
            # print("cell.shape ",cell.shape)
            # print("buffer.shape ",buffer.shape)
        
            #(buffer  != self.EOS.to(self.device)).all() and 
            while  len(out_seq) < out_max_Len :
                # print(buffer.view(1,1,-1).shape)
                
                
                out , (hidden , cell) = self.decoder(buffer.view(1,1,-1) , hidden , cell) 
                states += [(hidden , cell)]
                out_seq   += [out]
                out        = heapq.nlargest(1, enumerate( out[0] ) , key = lambda x : x[1])[0]
                
                if target != None and rd.random() < force_target_input_rate :
                    # print("ctd : " , ctd )
                    word   = self.embedding.itos[target[ctd]]  
                else:
                    word   = self.embedding.itos[ out[0] ]
            
                buffer = self.embedding[ word ].float().to(self.device) 
                ctd   += 1
            return torch.cat(out_seq , dim =0 ) , states   
        else :
            # while  len(out_seq) < out_max_Len :

            out , _ = self.decoder( master_Imput.view(1,1,-1) , hidden , cell) 
            out_seq   += [out]
            # out        = heapq.nlargest(1, enumerate( out[0] ) , key = lambda x : x[1])[0]
            
            # if target != None and rd.random() < force_target_input_rate :
            #     word   = self.embedding.itos[target[ctd]]  
            # else:
            #     word   = self.embedding.itos[ out[0] ]
        
            # buffer = self.embedding[ word ].float().to(self.device) 
            # ctd   += 1
        
            return torch.cat(out_seq , dim =0 ) , (hidden , cell)
    
    # @overload
    def fit(self , input_Batch :list , target_Batch:list , n , maxErro , maxAge = 1  , lossFunction = nn.CrossEntropyLoss() ,
            lossGraphNumber = 1 , test_Input_Batch = None , test_Target_Batch = None , out_max_Len = 150 , transform = None ) :

        optimizer = torch.optim.Adam(self.parameters(), n )
        lossValue = float("inf")
        Age = 0
        lossList = []
        bestLossValue = float("inf")
        # input_Batch = [i.view(1 , i.shape[0] , i.shape[1] ) for i in input_Batch ]    

        if test_Input_Batch != None and test_Target_Batch != None :
            lossTestList = []

        while lossValue > maxErro and Age < maxAge :
            lossValue = 0
            ctd = 0
            print("Age atual {}".format(Age))
            for x,y in zip(input_Batch , target_Batch ) :
                if transform != None :
                    x , y = transform(x) , transform(y)
                if type(y) != type(torch.tensor([1])) :
                    x = torch.from_numpy(x).float()
                    y = torch.from_numpy(y).float()
                div = len(y)
                                
                out , _ = self.forward_fit(x ,out_max_Len = y.shape[0] ,target = y.to(self.device) )

                print("Age atual {} , ctd atual {}\nout.shape = {} , y.shape = {}".format(Age ,ctd ,out.shape , y.shape))
                loss = lossFunction(out , y.to(self.device))/div
                lossValue += loss.item()
                print("Pré backward")
                loss.backward()
                print("Pós backward")
                optimizer.step()
                optimizer.zero_grad()
                ctd += 1
            if test_Input_Batch != None and test_Target_Batch != None  :
                diff = 0
                div = min( len(test_Input_Batch) , len(test_Target_Batch) )
                for x,y in zip( test_Input_Batch , test_Target_Batch ) :
                    if transform != None :
                        x , y = transform(x) , transform(y)
                    if type(y) != type(torch.tensor([1])) :
                        x = torch.from_numpy(x).float()
                        y = torch.from_numpy(y).float()

                    _ , out , __ = self.forward(x.to(self.device) , out_max_Len = out_max_Len )
                    diff += diff_Rate(out , y.to(self.device) )
                    
                lossTestList += [diff/div]
                if  lossTestList[-1] < bestLossValue :
                    print("Novo melhor")
                    best_Encoder  =  cp.deepcopy(self.encoder)
                    best_Decoder  =  cp.deepcopy(self.decoder)
                    bestLossValue =  lossTestList[-1]
                    print("Saiu do Melhor")

            Age += 1
            lossValue = lossValue/len(target_Batch)
            lossList.append(lossValue)
        
        if test_Input_Batch != None and test_Target_Batch != None  :
            print("O melhor resultado de teste foi " , bestLossValue )
            self.encoder = cp.deepcopy(best_Encoder)
            self.decoder = cp.deepcopy(best_Decoder)
        
            trainLossPlot = plt.subplot(2,1,1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

            testLossPlot = plt.subplot(2,1,2)
            testLossPlot.plot(range(1 , Age + 1) , lossTestList )
            plt.ylabel("Test Percent Loss" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)
        else :
            trainLossPlot = plt.subplot(1 , 1 , 1)
            trainLossPlot.plot(range(1 , Age + 1) , lossList)
            plt.ylabel("Loss in Train" , fontsize = 14 )
            plt.xlabel("Ages" , fontsize = 14)

        if lossGraphNumber != 1 :
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/{}_BiLSTM_LossInTrain_Plot.png".format(lossGraphNumber) )
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/{}_BiLSTM_LossInTrain_Plot.pdf".format(lossGraphNumber) )
        else :
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/BiLSTM_LossInTrain_Plot.png")
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/BiLSTM_LossInTrain_Plot.pdf")
        plt.show()

class BiLSTM_Attention(nn.Module ):#, override ):
    def __init__(self , input_dim , hidden_size_Encoder , num_Layers_Encoder ,
            hidden_size_Decoder , num_Layers_Decoder , num_classes , embedding , EOS_Vector ,device = torch.device("cpu"),
            attention_Shape = None , relu_Layer_Attention = False ):
        super(BiLSTM_Attention, self).__init__()
        
        self.input_dim   = input_dim
        
        self.hidden_size_Encoder = hidden_size_Encoder
        self.num_Layers_Encoder  = num_Layers_Encoder
        self.hidden_size_Decoder = hidden_size_Decoder
        self.num_Layers_Decoder  = num_Layers_Decoder
        self.attention_Shape     = None if attention_Shape == None else [2*hidden_size_Encoder*num_Layers_Encoder + 2*hidden_size_Decoder*num_Layers_Decoder] + list(attention_Shape) + [1]

        self.encoder   = Encoder( input_dim , hidden_size_Encoder , num_Layers_Encoder , device )
        self.decoder   = Decoder(  input_dim , hidden_size_Decoder , num_Layers_Decoder , num_classes , device)
        
        if attention_Shape == None :
            self.attention = nn.Linear(2*hidden_size_Encoder*num_Layers_Encoder + 2*hidden_size_Decoder*num_Layers_Decoder , 1 ).to(device)
        else :
            if relu_Layer_Attention == False :
                self.attention = nn.Sequential(*[nn.Linear(self.attention_Shape[i-1],self.attention_Shape[i]).to(device) for i in range(1,len(self.attention_Shape))])
            else :
                self.attention = []
                for i in range(1,len(self.attention_Shape) - 1) :
                    self.attention = self.attention + [nn.Linear(self.attention_Shape[i-1],self.attention_Shape[i]).to(device) , nn.ELU().to(device)]
                self.attention  = self.attention + [nn.Linear(self.attention_Shape[-2],self.attention_Shape[-1]).to(device) ]
                self.attention  = nn.Sequential(*self.attention)
        #    hidden_size_Decoder*num_Layers_Decoder*2 )
        self.device = device

        self.embedding = embedding
        self.EOS = EOS_Vector.to(device)
        self.BOS = -EOS_Vector.to(device)

    # def setDevice(self , device):
    #     self.device = device
    #     self.encoder.setDevice(device)
    #     self.decoder.setDevice(device)
    #     self.attention = self.attention.to(device)
    #     self.EOS = self.EOS.to(self.device)
    #     self.BOS = self.BOS.to(self.device)

    # def forward(self , x , out_max_Len = 150) :
    #     #ENCODER :
    #     hidden_State = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
    #     cell_State   = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )

    #     hidden_State , _ = self.encoder(x.view(1 , x.shape[0] , x.shape[1] ).to(self.device) , hidden_State , cell_State )

    #     hidden_State = hidden_State[0]


    #     #DECODER :
    #     out_class_Seq = []
    #     seq = []
    #     buffer = self.BOS.to(self.device)

    #     ctd    = 0
    #     hidden = torch.zeros(self.num_Layers_Decoder*2 , 1 , self.hidden_size_Decoder ,device = self.device )
    #     cell   = torch.zeros(self.num_Layers_Decoder*2 , 1 , self.hidden_size_Decoder ,device = self.device )

    #     while (buffer  != self.EOS.to(self.device)).all() and len(out_class_Seq) < out_max_Len :
    #         #ATTENTION :
    #         att_hidden  = self.attention( torch.cat((hidden_State , hidden.view(1 , -1).repeat(hidden_State.shape[0] , 1)) ,dim = 1 ) ) 
    #         att_hidden  = F.softmax(  att_hidden , dim = 0)
    #         att_hidden  = torch.einsum("ik,ij->j",(att_hidden,hidden_State))


    #         out , (hidden , cell) = self.decoder(buffer.view(1, 1 ,-1) , att_hidden.view(cell.shape[0],cell.shape[1],cell.shape[2]) , cell) 
            
    #         out            = heapq.nlargest(1, enumerate( out[0] ) , key = lambda x : x[1])[0]
            

    #         word   = self.embedding.itos[ out[0] ]
    #         buffer = self.embedding[ word ].float().to(self.device)

    #         seq           += [word] 
    #         out_class_Seq += [out[0]]
    #         ctd   += 1
            
    #     return seq , torch.tensor(out_class_Seq).to(self.device)

    # def forward_fit(self , x , out_max_Len = 150 ,target = None , force_target_input_rate = 0.5) :
        
    #     # x = x.view(1 , x.shape[0] , x.shape[1] )
    #     #ENCODER :
    #     hidden_State = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
    #     cell_State   = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
    #     # print("pré lista de estados")
    #     # for word in x.to(self.device) :
    #     #     hidden , cell = self.encoder(word.view(1 , 1 , word.shape[0] ) , hidden_State[-1] , cell_State[-1] )
    #     #     hidden_State += [hidden] 
    #     #     cell_State += [cell]
    #     hidden_State , _ = self.encoder(x.view(1 , x.shape[0] , x.shape[1] ).to(self.device) , hidden_State , cell_State )
    #     # hidden_State = hidden_State.permute(1,0,2)[0]
    #     # print("hidden_State.shape ",hidden_State.shape)
    #     # print("hidden_State.permute(1,0,2)[0].shape {}\nhidden_State[0].shape {}".format(hidden_State.permute(1,0,2)[0].shape , hidden_State[0].shape))
    #     hidden_State = hidden_State[0]
    #     # print("pós lista de estados")

    #     #DECODER :
    #     out_seq = []
    #     buffer = self.BOS.to(self.device)
    #     # print("self.BOS.shape = " , buffer.shape )
    #     ctd = 0
    #     hidden = torch.zeros(self.num_Layers_Decoder*2 , 1 , self.hidden_size_Decoder ,device = self.device )
    #     cell   = torch.zeros(self.num_Layers_Decoder*2 , 1 , self.hidden_size_Decoder ,device = self.device )
    #     # print("cell.shape = " , cell.shape )
    #     # (buffer  != self.EOS.to(self.device)).all() and 
    #     while len(out_seq) < out_max_Len :
    #         # print(buffer.view(1,1,-1).shape)
            
    #         #ATTENTION :
    #         # att_hidden = sum( self.attention(torch.cat((i.view(1 , -1 ) , hidden.view(1 , -1 ) ) , dim = 1) ).view(self.num_Layers_Decoder*2 , 1 ,self.input_dim) for i in hidden_State )
    #         # att_cell   = sum( self.attention(torch.cat((i.view(1 , -1 ) , cell.view(1 , -1 ) ) , dim = 1) ).view(self.num_Layers_Decoder*2 , 1 ,self.input_dim) for i in cell_State )
    #         # print("hidden_State.shape = " , hidden_State.shape)
    #         att_hidden  = self.attention( torch.cat((hidden_State , hidden.view(1 , -1).repeat(hidden_State.shape[0] , 1)) ,dim = 1 ) ) 
    #         # att_cell    = self.attention( torch.cat((cell_State , cell.view(1 , -1).repeat(cell_State.shape[0] , 1)) ,dim = 1 ) ) 
    #         # print(att_hidden[0])
    #         # print("pré SoftMax")
    #         att_hidden = F.softmax(  att_hidden , dim = 0)
            
    #         # att_cell   = F.softmax( att_cell  , dim = 0)
    #         # print("pos softmax hidden_State.shape {}".format(hidden_State.shape))
    #         # print("pos softmax att_hidden.shape {}".format(att_hidden.shape))
    #         # raise RuntimeError("Só pausando a execução , não tem erro nenhum aqui")
    #         # print("Pré attention att_hidden.shape = " , att_hidden.shape )

    #         # att_hidden  = sum( att_hidden[i]*hidden_State[i]  for i in range(len(hidden_State)))
    #         att_hidden = torch.einsum("ik,ij->j",(att_hidden,hidden_State)) 

    #         # att_cell    = sum( att_cell[i]*cell_State[i]  for i in range(len(cell_State)) )

    #         # print("Pós attention att_hidden.shape = " , att_hidden.shape )
    #         # print("cell.shape = " , cell.shape )
    #         # print( att_hidden[0] )
    #         # raise RuntimeError("Pausa rápida")
    #         # out , (hidden , cell) = self.decoder(buffer.view(1,1,-1) , att_hidden , cell)
    #         out , (hidden , cell) = self.decoder(buffer.view(1, 1 ,-1) , att_hidden.view(cell.shape[0],cell.shape[1],cell.shape[2]) , cell) 
    #         out_seq   += [out]
    #         out        = heapq.nlargest(1, enumerate( out[0] ) , key = lambda x : x[1])[0]
            
    #         if target != None and rd.random() < force_target_input_rate :
    #             word   = self.embedding.itos[target[ctd]]  
    #         else:
    #             word   = self.embedding.itos[ out[0] ]
        
    #         buffer = self.embedding[ word ].float().to(self.device) 
    #         ctd   += 1
            
        
    #     return torch.cat(out_seq , dim =0 )
    
    # def __saveLossGraph(self , path2Save :str  , Age : int , lossList : list , bestLossValue : float = None ,
    #     lossTestList : list = None ):
    #     if test_Input_Batch != None and test_Target_Batch != None  :
    #         print("O melhor resultado de teste foi " , bestLossValue )
    #         self.encoder = cp.deepcopy(best_Encoder)
    #         self.decoder = cp.deepcopy(best_Decoder)
        
    #         trainLossPlot = plt.subplot(2,1,1)
    #         trainLossPlot.plot(range(1 , Age + 1) , lossList)
    #         plt.ylabel("Loss in Train" , fontsize = 14 )
    #         plt.xlabel("Ages" , fontsize = 14)

    #         testLossPlot = plt.subplot(2,1,2)
    #         testLossPlot.plot(range(1 , Age + 1) , lossTestList )
    #         plt.ylabel("Test Percent Loss" , fontsize = 14 )
    #         plt.xlabel("Ages" , fontsize = 14)
    #     else :
    #         trainLossPlot = plt.subplot(1 , 1 , 1)
    #         trainLossPlot.plot(range(1 , Age + 1) , lossList)
    #         plt.ylabel("Loss in Train" , fontsize = 14 )
    #         plt.xlabel("Ages" , fontsize = 14)

    #     if path2Save != None and test_Input_Batch != None and test_Target_Batch != None :
    #         plt.savefig(f"{path2Save}_BiLSTM_ATTENTON_LossInTrain_Plot.png" )
    #         plt.savefig(f"{path2Save}_BiLSTM_ATTENTON_LossInTrain_Plot.pdf" )
    #     else :
    #         plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.png")
    #         plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.pdf")
    
    
    # def train_Step(self ,input_Batch :list , target_Batch : list , optimizer , lossFunction ,bestLossValue : float ,
    #         ctd : int , lossValue : int , test_Input_Batch= None , test_Target_Batch = None ,  out_max_Len = 150 ,
    #         best_Encoder = None , best_Decoder  = None , lossTestList = [] , transform = None ) :
    #     for x,y in zip(input_Batch , target_Batch ) :
    #         if transform != None :
    #             x , y = transform(x) , transform(y)
    #         if type(y) != type(torch.tensor([1])) :
    #             x = torch.from_numpy(x).float()
    #             y = torch.from_numpy(y).float()
    #         div = len(y)
                            
    #         out = self.forward_fit(x , out_max_Len = y.shape[0] ,target = y.to(self.device) )

    #         print("Age atual {} , ctd atual {}\nout.shape = {} , y.shape = {}".format(Age ,ctd ,out.shape , y.shape))
    #         loss = lossFunction(out , y.to(self.device))/div
    #         lossValue += loss.item()
    #         print("Pré backward")
    #         loss.backward()
    #         print("Pós backward")
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         ctd += 1
    #     if test_Input_Batch != None and test_Target_Batch != None and best_Encoder != None and best_Decoder != None  :
    #         diff = 0
    #         div = min( len(test_Input_Batch) , len(test_Target_Batch) )
    #         for x,y in zip( test_Input_Batch , test_Target_Batch ) :
    #             if transform != None :
    #                 x , y = transform(x) , transform(y)
    #             if type(y) != type(torch.tensor([1])) :
    #                 x = torch.from_numpy(x).float()
    #                 y = torch.from_numpy(y).float()

    #             _ , out = self.forward(x.to(self.device) , out_max_Len = out_max_Len )
    #             diff += diff_Rate(out , y.to(self.device) )
                
    #         lossTestList += [diff/div]
    #         if  lossTestList[-1] < bestLossValue :
    #             print("Novo melhor")
    #             best_Encoder  =  cp.deepcopy(self.encoder) 
    #             best_Decoder  =  cp.deepcopy(self.decoder)
    #             bestLossValue =  lossTestList[-1]
    #             print("Saiu do Melhor")
        
    #     if test_Input_Batch != None and test_Target_Batch != None  :
    #         return best_Encoder , best_Decoder , lossValue , lossTestList
    #     else :
    #         return _ , _ , lossValue , _
    
    
    # @overload #FALTA IMPLEMENTAR A PARTE DO SPARK
    # def fit(self , train_Batch_Path : str , n , maxErro , maxAge = 1 , lossFunction = nn.CrossEntropyLoss() ,
    #         rows_by_Step = 50 ,lossGraphPath = None , test_Batch_Path = None , out_max_Len = 150 , transform = None ) :
    #     """Usa PySpark Pra iterar ao longo do DataSet em formato csv """
    #     optimizer = torch.optim.Adam(self.parameters(), n )
    #     lossValue = float("inf")
    #     Age = 0
    #     lossList = []
    #     lossTestList = []
    #     bestLossValue  = float("inf")
    #     spark = SparkSession.builder.getOrCreate()
    #     df = spark.read.csv(train_Batch_Path , header = True )
    #     if test_Batch_Path != None :
    #         df_test = spark.read.csv(test_Batch_Path , header = True )
    #         data_itr_test = df_test.rdd.toLocalIterator()
    #         test_size = df_test.count()

    #     data_itr = df.rdd.toLocalIterator()
    #     train_size = df.count()

    #     # input_Batch = [i.view(1 , i.shape[0] , i.shape[1] ) for i in input_Batch ]    


    #     while lossValue > maxErro and Age < maxAge :
    #         lossValue = 0
    #         ctd = 0
    #         print("Age atual {}".format(Age))
    #         #SPARK

    #         for input_Batch , target_Batch in data_itr : 
    #             best_Encoder , best_Decoder , lossValue , lossTestList = self.train_Step(input_Batch , target_Batch , optimizer ,
    #             lossFunction ,bestLossValue ,ctd ,lossValue , test_Input_Batch , test_Target_Batch , out_max_Len , lossTestList ,
    #             transform )
    #         """for x,y in zip(input_Batch , target_Batch ) :
    #             if type(y) != type(torch.tensor([1])) :
    #                 x = torch.from_numpy(x).float()
    #                 y = torch.from_numpy(y).float()
    #             div = len(y)
                                
    #             out = self.forward_fit(x , out_max_Len = y.shape[0] ,target = y.to(self.device) )

    #             print("Age atual {} , ctd atual {}\nout.shape = {} , y.shape = {}".format(Age ,ctd ,out.shape , y.shape))
    #             loss = lossFunction(out , y.to(self.device))/div
    #             lossValue += loss.item()
    #             print("Pré backward")
    #             loss.backward()
    #             print("Pós backward")
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             ctd += 1
    #         if test_Input_Batch != None and test_Target_Batch != None  :
    #             diff = 0
    #             div = min( len(test_Input_Batch) , len(test_Target_Batch) )
    #             for x,y in zip( test_Input_Batch , test_Target_Batch ) :
    #                 if type(y) != type(torch.tensor([1])) :
    #                     x = torch.from_numpy(x).float()
    #                     y = torch.from_numpy(y).float()

    #                 _ , out = self.forward(x.to(self.device) , out_max_Len = out_max_Len )
    #                 diff += diff_Rate(out , y.to(self.device) )
                    
    #             lossTestList += [diff/div]
    #             if  lossTestList[-1] < bestLossValue :
    #                 print("Novo melhor")
    #                 best_Encoder  =  cp.deepcopy(self.encoder)
    #                 best_Decoder  =  cp.deepcopy(self.decoder)
    #                 bestLossValue =  lossTestList[-1]
    #                 print("Saiu do Melhor")"""

    #         Age += 1
    #         lossValue = lossValue/len(target_Batch)
    #         lossList.append(lossValue)
        
    #     if test_Input_Batch != None and test_Target_Batch != None  :
    #         print("O melhor resultado de teste foi " , bestLossValue )
    #         self.encoder = cp.deepcopy(best_Encoder)
    #         self.decoder = cp.deepcopy(best_Decoder)
        
    #     self.__saveLossGraph(lossGraphPath  , Age  , lossList  , bestLossValue , lossTestList)
    #     """if test_Input_Batch != None and test_Target_Batch != None  :
    #         print("O melhor resultado de teste foi " , bestLossValue )
    #         self.encoder = cp.deepcopy(best_Encoder)
    #         self.decoder = cp.deepcopy(best_Decoder)
        
    #         trainLossPlot = plt.subplot(2,1,1)
    #         trainLossPlot.plot(range(1 , Age + 1) , lossList)
    #         plt.ylabel("Loss in Train" , fontsize = 14 )
    #         plt.xlabel("Ages" , fontsize = 14)

    #         testLossPlot = plt.subplot(2,1,2)
    #         testLossPlot.plot(range(1 , Age + 1) , lossTestList )
    #         plt.ylabel("Test Percent Loss" , fontsize = 14 )
    #         plt.xlabel("Ages" , fontsize = 14)
    #     else :
    #         trainLossPlot = plt.subplot(1 , 1 , 1)
    #         trainLossPlot.plot(range(1 , Age + 1) , lossList)
    #         plt.ylabel("Loss in Train" , fontsize = 14 )
    #         plt.xlabel("Ages" , fontsize = 14)

    #     if lossGraphPath != None and test_Input_Batch != None and test_Target_Batch != None :
    #         plt.savefig(f"{lossGraphPath}_BiLSTM_ATTENTON_LossInTrain_Plot.png" )
    #         plt.savefig(f"{lossGraphPath}_BiLSTM_ATTENTON_LossInTrain_Plot.pdf" )
    #     else :
    #         plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.png")
    #         plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.pdf")"""
    #     plt.show()

    # @overload
    # def fit(self , input_Batch :list , target_Batch : list, n , maxErro , maxAge = 1 , lossFunction = nn.CrossEntropyLoss() ,
    #         lossGraphPath = None , test_Input_Batch = None, test_Target_Batch = None , out_max_Len  = 150 , transform = None) :

    #     optimizer = torch.optim.Adam(self.parameters(), n )
    #     lossValue = float("inf")
    #     Age = 0
    #     lossList = []
    #     bestLossValue = float("inf")
    #     # input_Batch = [i.view(1 , i.shape[0] , i.shape[1] ) for i in input_Batch ]    

    #     if test_Input_Batch != None and test_Target_Batch != None :
    #         lossTestList = []

    #     while lossValue > maxErro and Age < maxAge :
    #         lossValue = 0
    #         ctd = 0
    #         print("Age atual {}".format(Age))
    #         best_Encoder , best_Decoder , lossValue = self.train_Step(input_Batch , target_Batch , optimizer  ,
    #         lossFunction ,bestLossValue ,ctd ,lossValue , test_Input_Batch , test_Target_Batch , out_max_Len , transform )
    #         """for x,y in zip(input_Batch , target_Batch ) :
    #             if type(y) != type(torch.tensor([1])) :
    #                 x = torch.from_numpy(x).float()
    #                 y = torch.from_numpy(y).float()
    #             div = len(y)
                                
    #             out = self.forward_fit(x , out_max_Len = y.shape[0] ,target = y.to(self.device) )

    #             print("Age atual {} , ctd atual {}\nout.shape = {} , y.shape = {}".format(Age ,ctd ,out.shape , y.shape))
    #             loss = lossFunction(out , y.to(self.device))/div
    #             lossValue += loss.item()
    #             print("Pré backward")
    #             loss.backward()
    #             print("Pós backward")
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             ctd += 1
    #         if test_Input_Batch != None and test_Target_Batch != None  :
    #             diff = 0
    #             div = min( len(test_Input_Batch) , len(test_Target_Batch) )
    #             for x,y in zip( test_Input_Batch , test_Target_Batch ) :
    #                 if type(y) != type(torch.tensor([1])) :
    #                     x = torch.from_numpy(x).float()
    #                     y = torch.from_numpy(y).float()

    #                 _ , out = self.forward(x.to(self.device) , out_max_Len = out_max_Len )
    #                 diff += diff_Rate(out , y.to(self.device) )
                    
    #             lossTestList += [diff/div]
    #             if  lossTestList[-1] < bestLossValue :
    #                 print("Novo melhor")
    #                 best_Encoder  =  cp.deepcopy(self.encoder)
    #                 best_Decoder  =  cp.deepcopy(self.decoder)
    #                 bestLossValue =  lossTestList[-1]
    #                 print("Saiu do Melhor")"""

    #         Age += 1
    #         lossValue = lossValue/len(target_Batch)
    #         lossList.append(lossValue)
        
    #     if test_Input_Batch != None and test_Target_Batch != None  :
    #         print("O melhor resultado de teste foi " , bestLossValue )
    #         self.encoder = cp.deepcopy(best_Encoder)
    #         self.decoder = cp.deepcopy(best_Decoder)
        
    #     self.__saveLossGraph(lossGraphPath  , Age  , lossList  , bestLossValue , lossTestList)
    #     """    trainLossPlot = plt.subplot(2,1,1)
    #         trainLossPlot.plot(range(1 , Age + 1) , lossList)
    #         plt.ylabel("Loss in Train" , fontsize = 14 )
    #         plt.xlabel("Ages" , fontsize = 14)

    #         testLossPlot = plt.subplot(2,1,2)
    #         testLossPlot.plot(range(1 , Age + 1) , lossTestList )
    #         plt.ylabel("Test Percent Loss" , fontsize = 14 )
    #         plt.xlabel("Ages" , fontsize = 14)
    #     else :
    #         trainLossPlot = plt.subplot(1 , 1 , 1)
    #         trainLossPlot.plot(range(1 , Age + 1) , lossList)
    #         plt.ylabel("Loss in Train" , fontsize = 14 )
    #         plt.xlabel("Ages" , fontsize = 14)

    #     if lossGraphPath != None and test_Input_Batch != None and test_Target_Batch != None :
    #         plt.savefig(f"{lossGraphPath}_BiLSTM_ATTENTON_LossInTrain_Plot.png" )
    #         plt.savefig(f"{lossGraphPath}_BiLSTM_ATTENTON_LossInTrain_Plot.pdf" )
    #     else :
    #         plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.png")
    #         plt.savefig("BiLSTM_ATTENTON_LossInTrain_Plot.pdf")"""
    #     plt.show()




# class attention_Layer(nn.Module):
#     def __init__(self , shape , device = torch.device("cpu")):
#         super(attention_Layer , self ).__init__()
#         self.layer = nn.ModuleList([nn.Linear(shape[i-1],shape[i]).to(device) for i in range(1,len(shape))])
#         self.device = device
#     def set_Device(self , device):
#         for i in range(len(self.layer)) :
#             self.layer[i] = self.layer[i].to(device)
#         self.device = device
#     def forward(self , x ):
#         for i in self.layer :
#             x = i(x)
#         return x

class master_Slave_Encode_Decoder(nn.Module) :
    def __init__(self , slaves , input_dim , hidden_size_Encoder , num_Layers_Encoder ,
            hidden_size_Decoder , num_Layers_Decoder  , embedding , EOS_Vector ,device = torch.device("cpu"),
            attention_Shape = None , relu_Layer_Attention = False ):
        super(master_Slave_Encode_Decoder , self ).__init__()
        self.device = device
        
        self.slaves = []
        for i in slaves :
            self.slaves += [i]
        

        
        self.master_Encoder = BiLSTM(input_dim , hidden_size_Encoder , num_Layers_Encoder , hidden_size_Decoder ,
            num_Layers_Decoder , len(self.slaves) + 2 , embedding , EOS_Vector , device = device )
        self.slaves = nn.ModuleList(self.slaves)
        # self.encoderTransLayer = nn.Linear(input_dim * hidden_size_Encoder * num_Layers_Encoder * 2 , input_dim )

    def setDevice(self , device):
        self.device = device
        self.master_Encoder.setDevice(device)
        for i in slaves :
            i.setDevice(device)
        
    def forward(self ,x , out_max_Len = 150 ):

        # seq , out_class_Seq , states = self.master_Encoder(x , out_max_Len = 150 )
        # for i in range( len( out_class_Seq )) :
        #     if out_class_Seq[i] != len()
        pass

    def forward_fit(self , x , target  , out_max_Len = 150 , force_target_input_rate = 0.5 ,force_master_out_rate = .5 ):
        out_seq , states = self.master_Encoder.forward_fit(x , out_max_Len , target ,  force_target_input_rate)
        slave_out = []
        for i in range( len(target) ) :
            
            # out   =  heapq.nlargest( 1 , enumerate( out_seq[i] ) , key = lambda x : x[1])[0]
            print("target[i] : " ,target[i])
            print("i         : " ,i)
            # print("states[i][0] : " , states[i][0].shape)
            print("len(states) : " , len(states) )
            if target[i]<len(self.slaves) :
                out , _     =  self.slaves[ target[ i ] ].forward_fit( x , master_Imput = states[i][0] )
                slave_out  +=  [out.view(-1)]

            # if target != None and rd.random() < force_master_out_rate :
            #     out             = heapq.nlargest( 1 , enumerate( out_seq[i] ) , key = lambda x : x[1])[0]
            #     slave_seq , _   = self.slaves[ out[0] ].forward_fit( states[i] )

            # else :
            
        out_seq   = [i for i in out_seq ]
        return out_seq , slave_out 
    
    def fit(self , x , y_Master , y_Slave , n , maxErro , maxAge = 1  , lossFunction = nn.CrossEntropyLoss() ):
        #                           [i.parameters() for i in self.slaves ]+[self.master_Encoder.parameters()]
        optimizer  = torch.optim.Adam( self.parameters()  , n )
        Age = 0
        lossValue = float("inf")
        bestValue = float("inf")
        lossList  = []
        while Age < maxAge and lossValue > maxErro :
            lossValue = 0
            for x_In , y_Mas , y_Sla in zip(x , y_Master , y_Slave):
                y   = torch.cat([y_Mas , y_Sla] , dim = 1).permute(1,0).view(-1)
                div = len(y)

                
                out_Master , out_Slave = self.forward_fit(x_In , y_Mas.view(-1)  , len(y_Mas.view(-1)) )
                #c = pad_sequence([a.view(-1) , b.view(-1) , d.view(-1)]).permute(1,0)
                
                out = pad_sequence(out_Master + out_Slave).permute(1,0)
                # y   = pad_sequence(y ).permute(1,0)

                print("out.shape = {}\ny.shape = {} ".format(out.shape , y.shape ))
                loss       = lossFunction(out , y.to(self.device) )/div
                lossValue += loss.item()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            Age += 1
            lossValue = lossValue / min(len(x) , len(y_Master) , len(y_Slave) )
            lossList += [lossValue]
            if lossValue > bestValue :
                best_Slaves = cp.deepcopy(self.slaves)
                best_Master = cp.deepcopy(self.master_Encoder)
        self.slaves         = best_Slaves
        self.master_Encoder = best_Master

        plt.plot(range(len(lossList)) , lossList)
        plt.title("Erro")
        plt.xlabel("Ages" , fontsize = 14)
        plt.show()

