import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import random as rd
from matplotlib import pyplot as plt
print("teste")
class Encoder(nn.Module):
    def __init__(self , input_dim , hidden_size , num_Layers  ):
        super(Encoder , self).__init__()
        self.hidden_size = hidden_size
        self.input_dim   = input_dim
        self.num_Layers  = num_Layers
        self.lstm        = nn.LSTM(input_dim , hidden_size , num_Layers , batch_first = True , bidirectional = True)
        # self.linear      = nn.Linear(hidden_size*2 , num_classes )

    def forward(self , x , hidden , cell ) :
        # h0 = torch.zeros(self.num_Layers*2 , len(x) , self.hidden_size )
        # c0 = torch.zeros(self.num_Layers*2 , len(x) , self.hidden_size )
        
        x , (hidden , cell )  = self.lstm(x , (hidden , cell))

        # return self.lstm(x , (h0 , c0))[1]
        return hidden , cell

class Decoder(nn.Module):
    def __init__(self , input_dim , hidden_size , num_Layers , num_classes ):
        super(Decoder , self).__init__()
        self.hidden_size = hidden_size
        self.input_dim   = input_dim
        self.num_Layers  = num_Layers
        self.lstm        = nn.LSTM(input_dim , hidden_size , num_Layers , batch_first = True , bidirectional = True)
        self.linear      = nn.Linear(hidden_size*2 , num_classes )

    def forward(self , x , hidden_State , cell_State) :
        # h0 = torch.zeros(self.num_Layers*2 , x.size(0) , self.hidden_size )
        # c0 = torch.zeros(self.num_Layers*2 , x.size(0) , self.hidden_size )
        
        x , (hidden_State , cell_State) = self.lstm(x , (hidden_State , cell_State))

        return self.linear(x[ : , -1 , : ]) , (hidden_State , cell_State)

class BiLSTM(nn.Module):
    def __init__(self , input_dim , hidden_size_Encoder , num_Layers_Encoder ,
            hidden_size_Decoder , num_Layers_Decoder , num_classes , embedding , EOS_Vector ):
        super(BiLSTM , self).__init__()
        
        self.input_dim   = input_dim
        
        self.hidden_size_Encoder = hidden_size_Encoder
        self.num_Layers_Encoder  = num_Layers_Encoder
        self.hidden_size_Decoder = hidden_size_Decoder
        self.num_Layers_Decoder  = num_Layers_Decoder

        self.encoder   = Encoder( input_dim , hidden_size_Encoder , num_Layers_Encoder)
        self.decoder   = Decoder(  input_dim , hidden_size_Decoder , num_Layers_Decoder , num_classes)
        self.attention = nn.Linear(2*hidden_size_Encoder*num_Layers_Encoder + 2*hidden_size_Decoder*num_Layers_Decoder , 1 )
        #    hidden_size_Decoder*num_Layers_Decoder*2 )
        self.embedding = embedding
        self.EOS = EOS_Vector
        self.BOS = -EOS_Vector

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
        hidden_State = [torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder )]
        cell_State   = [torch.zeros(self.num_Layers_Encoder*2 , 1 , self.hidden_size_Encoder )]
        print("pré lista de estados")
        for word in x :
            hidden , cell = self.encoder(word.view(1 , 1 , word.shape[0] ) , hidden_State[-1] , cell_State[-1] )
            hidden_State += [hidden] 
            cell_State += [cell]
        print("pós lista de estados")
        
        #DECODER :
        out_seq = []
        buffer = self.BOS.view(1,1,-1)
        ctd = 0
        hidden = torch.zeros(self.num_Layers_Decoder*2 , 1 , self.hidden_size_Decoder )
        cell   = torch.zeros(self.num_Layers_Decoder*2 , 1 , self.hidden_size_Decoder )

        while (buffer  != self.EOS).all() and len(out_seq) < out_max_Len :
            # print(buffer.view(1,1,-1).shape)
            
            #ATTENTION :
            # att_hidden = sum( self.attention(torch.cat((i.view(1 , -1 ) , hidden.view(1 , -1 ) ) , dim = 1) ).view(self.num_Layers_Decoder*2 , 1 ,self.input_dim) for i in hidden_State )
            # att_cell   = sum( self.attention(torch.cat((i.view(1 , -1 ) , cell.view(1 , -1 ) ) , dim = 1) ).view(self.num_Layers_Decoder*2 , 1 ,self.input_dim) for i in cell_State )
            att_hidden  = tuple((self.attention(torch.cat((i.view( -1 ) , hidden.view( -1 ) ) , dim = 0))  for i in hidden_State)) 
            att_cell    = tuple((self.attention(torch.cat((i.view( -1 ) , cell.view( -1 ) ) , dim = 0) )  for i in cell_State ))
            # print(att_hidden[0])
            print("pré SoftMax")
            att_hidden = F.softmax( torch.cat( att_hidden , dim = 0 ) , dim = 0)
            att_cell   = F.softmax( torch.cat( att_cell , dim = 0 ) , dim = 0)
            print("pos softmax")
            att_hidden  = sum( att_hidden[i]*hidden_State[i]  for i in range(len(hidden_State)))
            att_cell    = sum( att_cell[i]*cell_State[i]  for i in range(len(cell_State)) )

            
            out , (hidden , cell) = self.decoder(buffer.view(1,1,-1) , att_hidden , att_cell) 
            out_seq   += [out]
            out        = heapq.nlargest(1, enumerate( buffer ) , key = lambda x : x[1])[0]
            
            if target != None and rd.random() < force_target_input_rate :
                word   = self.embedding.index2word[target[ctd]]  
            else:
                word   = self.embedding.index2word[ out[0] ]
        
            buffer = torch.from_numpy( self.embedding[ word ] ).float() 
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
                                
                out = self.forward_fit(x ,out_max_Len = y.shape[0] ,target = y )

                print("Age atual {} , ctd atual {}\nout.shape = {} , y.shape = {}".format(Age ,ctd ,out.shape , y.shape))
                loss = lossFunction(out , y)/div
                lossValue += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ctd += 1
            Age += 1
            lossValue = lossValue/len(target_Batch)
            lossList.append(lossValue)
        
        plt.plot(range(1 , Age + 1) , lossList)
        if lossGraphNumber != 1 :
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/{}_BiLSTM_LossInTrain_Plot.png".format(lossGraphNumber) )
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/{}_BiLSTM_LossInTrain_Plot.pdf".format(lossGraphNumber) )
        else :
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/BiLSTM_LossInTrain_Plot.png")
            plt.savefig("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/BiLSTM_LossInTrain_Plot.pdf")
        plt.show()