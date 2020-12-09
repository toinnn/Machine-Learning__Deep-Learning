import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import random as rd
from matplotlib import pyplot as plt
import copy as cp
print("teste")
def diff_Rate(a,b):
    smallerSize = min(len(a) , len(b))
    correct = 0
    for i in range(smallerSize):
        correct += 1 if a[i]==b[i] else 0
    biggerSize = max(len(a) , len(b))
    return 1 - correct/(biggerSize) 

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

class BiLSTM(nn.Module):
    def __init__(self , input_dim , hidden_size_Encoder , num_Layers_Encoder ,
            hidden_size_Decoder , num_Layers_Decoder , num_classes , embedding , EOS_Vector , device = torch.device("cpu") ):
        super(BiLSTM , self).__init__()
        
        self.input_dim   = input_dim
        self.device      = device
        
        self.hidden_size_Encoder = hidden_size_Encoder
        self.num_Layers_Encoder  = num_Layers_Encoder
        self.hidden_size_Decoder = hidden_size_Decoder 
        self.num_Layers_Decoder  = num_Layers_Decoder

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

    def forward(self , x , out_max_Len = 150) :
        #ENCODER :
        hidden_State = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
        cell_State   = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
        
        _ , (hidden , cell) = self.encoder(x.view(1 , x.shape[0] , x.shape[1] ).to(self.device) , hidden_State , cell_State )
        

        #DECODER :
        seq = []
        out_class_Seq = []
        buffer = self.BOS.view(1,1,-1).to(self.device)
        ctd = 0
        while (buffer  != self.EOS.to(self.device)).all() and len(seq) < out_max_Len :
            
            out , (hidden , cell) = self.decoder(buffer.view(1,1,-1) , hidden , cell) 
            out        = heapq.nlargest( 1 , enumerate( buffer ) , key = lambda x : x[1] )[0]

            word   = self.embedding.itos[ out[0] ]
            buffer = self.embedding[ word ].float().to(self.device)

            seq           += [word] 
            out_class_Seq += [out[0]]
            ctd   += 1
            
        return seq , torch.tensor(out_class_Seq).to(self.device)
        # return self.linear(x[ : , -1 , : ])

    def forward_fit(self , x , out_max_Len = 150 ,target = None , force_target_input_rate = 0.5) :
        
        # x = x.view(1 , x.shape[0] , x.shape[1] )
        #ENCODER :
        hidden_State = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
        cell_State   = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
        
        _ , (hidden , cell) = self.encoder(x.view(1 , x.shape[0] , x.shape[1] ).to(self.device) , hidden_State , cell_State )
        
        
        #DECODER :
        out_seq = []
        buffer = self.BOS.view(1,1,-1).to(self.device)
        ctd = 0
        # teste = (buffer  != self.EOS.to(self.device)).all()
        # print("hidden.shape ",hidden.shape)
        # print("cell.shape ",cell.shape)
        # print("buffer.shape ",buffer.shape)

        while (buffer  != self.EOS.to(self.device)).all() and len(out_seq) < out_max_Len :
            # print(buffer.view(1,1,-1).shape)
            
            
            out , (hidden , cell) = self.decoder(buffer.view(1,1,-1) , hidden , cell) 
            out_seq   += [out]
            out        = heapq.nlargest(1, enumerate( buffer ) , key = lambda x : x[1])[0]
            
            if target != None and rd.random() < force_target_input_rate :
                word   = self.embedding.itos[target[ctd]]  
            else:
                word   = self.embedding.itos[ out[0] ]
        
            buffer = self.embedding[ word ].float().to(self.device) 
            ctd   += 1
            
        
        return torch.cat(out_seq , dim =0 )
    

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
                                
                out = self.forward_fit(x ,out_max_Len = y.shape[0] ,target = y.to(self.device) )

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
                div = min( len(input_Batch) , len(target_Batch) )
                for x,y in zip(input_Batch , target_Batch ) :
                    if type(y) != type(torch.tensor([1])) :
                        x = torch.from_numpy(x).float()
                        y = torch.from_numpy(y).float()

                    _ , out = self.forward(x.to(self.device) , out_max_Len = out_max_Len )
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

class BiLSTM_Attention(nn.Module):
    def __init__(self , input_dim , hidden_size_Encoder , num_Layers_Encoder ,
            hidden_size_Decoder , num_Layers_Decoder , num_classes , embedding , EOS_Vector ,device = torch.device("cpu"),
            attention_Shape = None ):
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
            self.attention = nn.Sequential(*[nn.Linear(attention_Shape[i-1],attention_Shape[i]).to(device) for i in range(1,len(attention_Shape))])
        #    hidden_size_Decoder*num_Layers_Decoder*2 )
        self.device = device

        self.embedding = embedding
        self.EOS = EOS_Vector.to(device)
        self.BOS = -EOS_Vector.to(device)

    def setDevice(self , device):
        self.device = device
        self.encoder.setDevice(device)
        self.decoder.setDevice(device)
        self.attention = self.attention.to(device)
        self.EOS = self.EOS.to(self.device)
        self.BOS = self.BOS.to(self.device)

    def forward(self , x , out_max_Len = 150) :
        # h0 = torch.zeros(self.num_Layers*2 , x.size(0) , self.hidden_size )
        # c0 = torch.zeros(self.num_Layers*2 , x.size(0) , self.hidden_size )
        
        hidden_State , cell_State = self.encoder(x)

        seq = []
        buffer = self.BOS
        while (buffer != self.EOS).all() and seq.shape[0] < out_max_Len :
            out , hidden_State , cell_State = self.decoder(buffer , hidden_State , cell_State)
            out    = heapq.nlargest(1, enumerate( buffer ) , key = lambda x : x[1])[0]
            word   = self.embedding.index2word[ out[0] ]
            buffer = torch.from_numpy( self.embedding[ word ] ).float()
            seq   += [word]
        
        return seq
        return self.linear(x[ : , -1 , : ])

    def forward_fit(self , x , out_max_Len = 150 ,target = None , force_target_input_rate = 0.5) :
        
        # x = x.view(1 , x.shape[0] , x.shape[1] )
        #ENCODER :
        hidden_State = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
        cell_State   = torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder ,device = self.device )
        # print("pré lista de estados")
        # for word in x.to(self.device) :
        #     hidden , cell = self.encoder(word.view(1 , 1 , word.shape[0] ) , hidden_State[-1] , cell_State[-1] )
        #     hidden_State += [hidden] 
        #     cell_State += [cell]
        hidden_State , _ = self.encoder(x.view(1 , x.shape[0] , x.shape[1] ).to(self.device) , hidden_State , cell_State )
        # hidden_State = hidden_State.permute(1,0,2)[0]
        # print("hidden_State.shape ",hidden_State.shape)
        # print("hidden_State.permute(1,0,2)[0].shape {}\nhidden_State[0].shape {}".format(hidden_State.permute(1,0,2)[0].shape , hidden_State[0].shape))
        hidden_State = hidden_State[0]
        # print("pós lista de estados")

        #DECODER :
        out_seq = []
        buffer = self.BOS.to(self.device)
        # print("self.BOS.shape = " , buffer.shape )
        ctd = 0
        hidden = torch.zeros(self.num_Layers_Decoder*2 , 1 , self.hidden_size_Decoder ,device = self.device )
        cell   = torch.zeros(self.num_Layers_Decoder*2 , 1 , self.hidden_size_Decoder ,device = self.device )
        # print("cell.shape = " , cell.shape )
        while (buffer  != self.EOS.to(self.device)).all() and len(out_seq) < out_max_Len :
            # print(buffer.view(1,1,-1).shape)
            
            #ATTENTION :
            # att_hidden = sum( self.attention(torch.cat((i.view(1 , -1 ) , hidden.view(1 , -1 ) ) , dim = 1) ).view(self.num_Layers_Decoder*2 , 1 ,self.input_dim) for i in hidden_State )
            # att_cell   = sum( self.attention(torch.cat((i.view(1 , -1 ) , cell.view(1 , -1 ) ) , dim = 1) ).view(self.num_Layers_Decoder*2 , 1 ,self.input_dim) for i in cell_State )
            # print("hidden_State.shape = " , hidden_State.shape)
            att_hidden  = self.attention( torch.cat((hidden_State , hidden.view(1 , -1).repeat(hidden_State.shape[0] , 1)) ,dim = 1 ) ) 
            # att_cell    = self.attention( torch.cat((cell_State , cell.view(1 , -1).repeat(cell_State.shape[0] , 1)) ,dim = 1 ) ) 
            # print(att_hidden[0])
            # print("pré SoftMax")
            att_hidden = F.softmax(  att_hidden , dim = 0)
            
            # att_cell   = F.softmax( att_cell  , dim = 0)
            # print("pos softmax hidden_State.shape {}".format(hidden_State.shape))
            # print("pos softmax att_hidden.shape {}".format(att_hidden.shape))
            # raise RuntimeError("Só pausando a execução , não tem erro nenhum aqui")
            # print("Pré attention att_hidden.shape = " , att_hidden.shape )

            # att_hidden  = sum( att_hidden[i]*hidden_State[i]  for i in range(len(hidden_State)))
            att_hidden = torch.einsum("ik,ij->j",(att_hidden,hidden_State)) 

            # att_cell    = sum( att_cell[i]*cell_State[i]  for i in range(len(cell_State)) )

            # print("Pós attention att_hidden.shape = " , att_hidden.shape )
            # print("cell.shape = " , cell.shape )
            # print( att_hidden[0] )
            # raise RuntimeError("Pausa rápida")
            # out , (hidden , cell) = self.decoder(buffer.view(1,1,-1) , att_hidden , cell)
            out , (hidden , cell) = self.decoder(buffer.view(1, 1 ,-1) , att_hidden.view(cell.shape[0],cell.shape[1],cell.shape[2]) , cell) 
            out_seq   += [out]
            out        = heapq.nlargest(1, enumerate( buffer ) , key = lambda x : x[1])[0]
            
            if target != None and rd.random() < force_target_input_rate :
                word   = self.embedding.itos[target[ctd]]  
            else:
                word   = self.embedding.itos[ out[0] ]
        
            buffer = self.embedding[ word ].float().to(self.device) 
            ctd   += 1
            
        
        return torch.cat(out_seq , dim =0 )
    

    def fit(self , input_Batch , target_Batch , n , maxErro , maxAge = 1 , lossFunction = nn.CrossEntropyLoss() ,
            lossGraphNumber = 1) :

        optimizer = torch.optim.Adam(self.parameters(), n )
        lossValue = float("inf")
        Age = 0
        lossList = []
        # input_Batch = [i.view(1 , i.shape[0] , i.shape[1] ) for i in input_Batch ]

        while lossValue > maxErro and Age < maxAge :
            lossValue = 0
            ctd = 0
            print("Age atual {}".format(Age))
            for x,y in zip(input_Batch , target_Batch ) :
                if type(y) != type(torch.tensor([1])) :
                    x = torch.from_numpy(x).float()
                    y = torch.from_numpy(y).float()
                div = len(y)
                                
                out = self.forward_fit(x , out_max_Len = y.shape[0] ,target = y.to(self.device) )

                print("Age atual {} , ctd atual {}\nout.shape = {} , y.shape = {}".format(Age ,ctd ,out.shape , y.shape))
                loss = lossFunction(out , y.to(self.device))/div
                lossValue += loss.item()
                print("Pré backward")
                loss.backward()
                print("Pós backward")
                optimizer.step()
                optimizer.zero_grad()
                ctd += 1
            Age += 1
            lossValue = lossValue/len(target_Batch)
            lossList.append(lossValue)
        
        plt.plot(range(1 , Age + 1) , lossList)
        if lossGraphNumber != 1 :
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/{}_BiLSTM_ATTENTON_LossInTrain_Plot.png".format(lossGraphNumber) )
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/{}_BiLSTM_ATTENTON_LossInTrain_Plot.pdf".format(lossGraphNumber) )
        else :
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/BiLSTM_ATTENTON_LossInTrain_Plot.png")
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/BiLSTM_ATTENTON_LossInTrain_Plot.pdf")
        plt.show()

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
