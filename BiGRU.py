import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import random as rd
from matplotlib import pyplot as plt
import copy as cp

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

class BiGRU(nn.Module):
    def __init__(self , input_dim , hidden_size_Encoder , num_Layers_Encoder ,
            hidden_size_Decoder , num_Layers_Decoder , num_classes , embedding , EOS_Vector , device = torch.device("cpu") ):
        super(BiGRU , self).__init__()
        
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
    

    def fit(self , input_Batch , target_Batch , n , maxErro , maxAge = 1  , lossFunction = nn.CrossEntropyLoss() ,
            lossGraphNumber = 1 , test_Input_Batch = None , test_Target_Batch = None , out_max_Len = 150  ) :

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
