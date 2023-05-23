import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random as rd
from torch.autograd import Variable
import heapq
# from skip_Gram import skip_gram

def oneHotEncode(dim, idx):
    vector = torch.zeros(dim)
    vector[idx] = 1.0
    return vector    
def mult_oneHotEncode(dim , idx) :
    return torch.cat( [ oneHotEncode(dim , i ).view(1,-1) for i in idx ]  , dim = 0 ).float()

def word2vec(wordVec , word ,dim):
    try:
        
        return torch.from_numpy(wordVec[word]).view(1,-1)
    except :
        print("Entrou no vetor não pertencente ao vocabulario ")
        return torch.ones([1,dim]).float()*-79

def sen2vec(gensimWorVec , sentence , dim) :
    sen = [token for token in re.split(r"(\W)", sentence ) if token != " " and token != "" ]
    print(sen)
    return torch.cat( [ word2vec(gensimWorVec , word ,dim ) for word in sen ] , dim = 0 )
    
def json2vec(js , key ,dim , gensimWorVec ):
    try :
        seq = [ vec for sen in js[key] for vec in (sen2vec(gensimWorVec , sen.lower() , dim ) , torch.ones([1,dim])*-127 ) ]
        seq.pop()
        return torch.cat( seq , dim = 0 )
    except :
        # Key não existente :
        print("Entrou no não existe a key ")
        return torch.ones([1,dim]).float()*-47
            


class selfAttention(nn.Module):
    def __init__(self , model_dim ,heads = 1):
        super(selfAttention , self ).__init__()
        assert model_dim % heads == 0 , "O número de heads deve ser um divisor do número de dimensões do vetor de Embedding "
        self.heads    = heads 
        self.head_dim = int(model_dim / heads)
        
        self.keys    = nn.ModuleList([ nn.Linear(model_dim , self.head_dim , bias = False)  for i in range(heads)])
        self.queries = nn.ModuleList([ nn.Linear(model_dim , self.head_dim , bias = False)  for i in range(heads)])
        self.values  = nn.ModuleList([ nn.Linear(model_dim , self.head_dim , bias = False)  for i in range(heads)])
        
        self.v = Variable(torch.rand(heads , self.head_dim , dtype = float) , requires_grad = True).float()
        self.u = Variable(torch.rand(heads , self.head_dim , dtype = float) , requires_grad = True).float()
        self.output  = nn.Linear(model_dim , model_dim )
        print("chegou aqui")

 
    def forward(self,value ,key , query, scale = True , mask = False ) :
        
        # if type(key) != type(torch.tensor([1])) :
        #     key   = torch.from_numpy(key)
        #     query = torch.from_numpy(query)
        #     value = torch.from_numpy(value)
        # key   = key.float()
        # query = query.float()
        # value = value.float()

        keys    = torch.cat([k(key  ) for k in self.keys    ],dim=0)
        queries = torch.cat([q(query) for q in self.queries ],dim=0)
        values  = torch.cat([v(value) for v in self.values  ],dim=0)
        
        keys    = torch.transpose(keys.reshape(self.heads ,key.shape[0], self.head_dim ) ,1,2)
        queries = queries.reshape(self.heads ,query.shape[0] , self.head_dim )
        values  = values.reshape(self.heads ,value.shape[0] , self.head_dim )


        #EMBEDDING Posicional Relativo :
        pos = torch.arange(0 , -keys.shape[2] , -1).view(1,1,-1)
        pos = torch.cat([pos +i for i in torch.arange( query.shape[0] )] , dim=1)#Faz somente a matriz t-j
        R   = torch.zeros(self.head_dim , query.shape[0] , key.shape[0])
        # print("pos.shape = {} \nR.shape = {}".format(pos.shape , R.shape))
        #             t.sin(t.cat(tuple(pos*(1e4**(-(2*i)*(1/self.head_dim))) for i in t.arange(pe[0::2,:,:].shape[0])),dim = 0).float())
        R[0::2,:,:] = torch.sin(torch.cat(tuple( pos*(1e4**(-2*i*(1/self.head_dim))) for i in torch.arange( R[0::2,:,:].shape[0] )) , dim = 0 ).float() )
        R[1::2,:,:] = torch.cos(torch.cat(tuple( pos*(1e4**(-2*i*(1/self.head_dim))) for i in torch.arange( int(R.shape[0]/2)) ) , dim = 0).float() )
        

        UxK = torch.cat([torch.cat([i for j in torch.arange(queries.shape[1])] , dim = 0)  for i in torch.einsum("hi,hik->hk",[self.u , keys])] , dim=0 )
        UxK = UxK.reshape(self.heads,queries.shape[1] , keys.shape[2])
        # print("v*R.shape = " , torch.einsum("hd,dtj->htj",[self.v,R]).shape )
        # # print("v.shape = {} ".format(self.v.shape))
        # print("R.shape = {} \nqueries.shape = {}".format(R.shape ,queries.shape )  )
        # # print("keys.shape = {} ".format(keys.shape) )
        # print("UxK.shape = " , UxK.shape )
        # print("queries*keys.shape = ", torch.einsum("lij,ljk->lik",[queries,keys]).shape )
        # print("queries*R = ", torch.einsum("hid,dij->hij",[queries,R]).shape )  #torch.einsum("pld,hlp->hld",[R,queries])
        attention = torch.einsum("lij,ljk->lik",[queries,keys]) + torch.einsum("hid,dij->hij",[queries,R]) + UxK + torch.einsum("hd,dtj->htj",[self.v,R]) 
        
        if mask :
            # attention = torch.einsum("lij,ljk->lik",[queries,keys])
            mask = torch.tril(torch.ones(attention.shape))
            if scale :
                attention = F.softmax(attention.masked_fill(mask==0, float("-1e18"))*(self.head_dim**-.5),dim = 2)
            else :
                attention = F.softmax(attention.masked_fill(mask==0, float("-1e18")),dim = 2)
        else :
            if scale :
                attention = F.softmax(attention*(self.head_dim**-.5),dim = 2)
            else :
                attention = F.softmax(attention , dim = 2)
        attention = torch.einsum("lij,ljk->lik" , [attention,values])
        attention = torch.cat( tuple(i for i in attention ) , 1 )

        # a = a.reshape(heads ,a_dim ,phrase_len)
        # a = t.transpose(a.reshape(heads ,phrase_len,a_dim) ,1,2)
        # b = b.reshape(heads ,phrase_len , b_dim)
        ## a = a.reshape(a_len,a_dim,1)
        ## b = b.reshape(b_len,1 ,b_dim)
        # print(t.einsum("lij,ljk->lik",[b,a]))
        #attention = F.softmax(t.einsum("lij,ljk->lik",[q,k])*(self.head_dim**-.5).float(),dim = 2)
        # attention = t.einsum("lij,ljk->lik",[attention,values])
        # attention = t.cat(tuple(i for i in attention),1)
        #return self.output(attention)
        
        return self.output(attention)
class transformerBlock(nn.Module):# A LAYER_NORM AINDA NÃO ME CONVENCEU
    def __init__(self, model_dim , heads , forward_expansion = 4):
        super(transformerBlock , self ).__init__()
        self.attention   = selfAttention(model_dim ,heads= heads)
        self.layerNorm0  = nn.LayerNorm(model_dim )
        self.layerNorm1  = nn.LayerNorm(model_dim )
        self.feedForward = nn.Sequential(nn.Linear(model_dim,model_dim * forward_expansion),nn.ELU(),nn.Linear(model_dim * forward_expansion ,model_dim))

    def forward(self ,value ,key ,query ,scale = True , mask = False):
        attention = self.attention(value ,key ,query , scale = scale , mask = mask)
        x = self.layerNorm0(attention + query)
        forward = self.feedForward(x)

        return self.layerNorm1(forward + x)
class encoder(nn.Module):
    def __init__(self ,model_dim , heads ,num_layers, forward_expansion = 4):
        super(encoder,self).__init__()
        self.blocks = nn.ModuleList([transformerBlock(model_dim , heads , forward_expansion = forward_expansion) for i in range(num_layers)])
        
    def forward(self,x , mask = False ,scale = True) :
        out = x
        for layer in self.blocks :
            out = layer(out ,out ,out , mask = mask , scale = scale)
        return out
class decoderBlock(nn.Module):
    def __init__(self ,model_dim ,heads ,forward_expansion = 4):
        super(decoderBlock,self).__init__()
        self.attention = selfAttention(model_dim , heads)
        self.norm = nn.LayerNorm(model_dim)
        self.transformerBlock = transformerBlock(model_dim,heads , forward_expansion = forward_expansion)

    def forward(self ,x ,values ,keys ,mask = True ,scale = False) :
        attention = self.attention(x,x,x , mask = mask , scale = scale)
        queries = self.norm(attention + x)
        return self.transformerBlock(values , keys , queries , scale = scale)
class decoder(nn.Module):
    def __init__(self,model_dim ,heads ,num_layers ,word_Embedding ,EOS , num_Classes  ,forward_expansion = 4):
        super(decoder,self).__init__()
        self.embedding = word_Embedding
        self.EOS = EOS  #End-Of-Sentence Vector
        self.BOS = -EOS #Begin-Of-Sentence Vector
        
        self.embedding["<BOS>"] = self.BOS#DEPOIS TIRAR ESSE NUMPY
        self.embedding["<EOS>"] = self.EOS#DEPOIS TIRAR ESSE NUMPY
        self.layers = nn.ModuleList( decoderBlock(model_dim , heads , forward_expansion = forward_expansion) for _ in torch.arange(num_layers))
        # self.linear_Out = nn.Linear(model_dim , len(self.embedding) )---OLHAR AKI DEPOIS---
        # self.linear_Out = nn.Linear(model_dim , len(self.embedding.vocab) )
        self.linear_Out = nn.Linear(model_dim , num_Classes )
        if type(EOS) != type(torch.tensor([1])) :
            self.EOS = torch.from_numpy(self.EOS).float()
            self.BOS = -self.EOS

    def forward_fit(self ,Enc_values , Enc_keys , max_lengh  ) :
        sequence = self.BOS
        soft_Out = [] # nn.ModuleList([])
        # if type(sequence) != type(torch.tensor([1])) :
        #     Enc_values = torch.from_numpy(Enc_values).float()
        #     Enc_keys   = torch.from_numpy(Enc_keys).float()
            # sequence   = torch.from_numpy(self.BOS).float()

        while  sequence.shape[0]<= max_lengh  :# Ta errado
            print("Mais um loop de Decoder e sequence.shape[0] = " , sequence.shape[0] )
            buffer = sequence
            for l in self.layers :
                buffer = l(buffer , Enc_values , Enc_keys)
            buffer = F.softmax(self.linear_Out(buffer[-1]) , dim = 0 )
            # out        = torch.argmax(buffer).item()
            out = heapq.nlargest(1, enumerate(buffer ) , key = lambda x : x[1])[0]
            soft_Out.append(buffer.view(1,-1))
            
            # sequence = torch.cat((sequence , torch.from_numpy(self.embedding[ self.embedding.index2word[ out ] ] ).float().view(1,-1)),dim = 0 )
            # sequence = torch.cat((sequence , self.embedding.vocabulary[self.embedding.idx2token[out[0]]]),dim = 0 )
            sequence = torch.cat((sequence , torch.from_numpy(self.embedding[ self.embedding.index2word[ out[0] ] ] ).float().view(1,-1)),dim = 0 )

        return torch.cat(soft_Out ,dim = 0)
        

    def forward(self ,Enc_values , Enc_keys , max_lengh = 100 ) :
        sequence = self.BOS
        idx = [len(self.embedding.vocabulary) - 2]
        while sequence[-1] != self.EOS and sequence.shape[0]< max_lengh  :# Ta errado
            buffer = sequence
            for l in layers :
                buffer = l(buffer , Enc_values , Enc_keys)
            buffer = F.softmax(self.linear_Out(buffer[-1]) , dim = 0 )
            #out        = torch.argmax(buffer).item()
            out = heapq.nlargest(1, enumerate(buffer ) , key = lambda y : y[1])[0]
            
            # idx.append(out)
            idx.append(out[0])
            #buffer = F.softmax(buffer , dim = 1)
            #buffer = O Vetor com a maior probabilidade , mas qual ??
            
            sequence = torch.cat((sequence , self.embedding.vocabulary[self.embedding.idx2token[out[0]]]),dim = 0 )
        sequence = [self.embedding.idx2token[i] for i in idx ]
        return sequence

        if sequence.shape[0] == max_lengh -1 :
            return torch.cat((sequence,self.EOS),dim = 0)
        return sequence

class Tener(nn.Module):#EOS_Vector == End-Of-Sentence_Vector 
    def __init__(self , model_dim ,heads_Enc , heads_Dec ,num_Enc_layers ,num_Dec_layers ,Embedding ,EOS_Vector , num_class ):
        super(Tener,self).__init__()
        self.model_dim = model_dim
        self.Embedding = Embedding
        self.encoder = encoder(model_dim , heads_Enc , num_Enc_layers)
        self.decoder = decoder(model_dim , heads_Dec , num_Dec_layers , Embedding , EOS_Vector , num_class )
        # self.optimizer = torch.optim.Adam(self.parameters(),0.05,(0.9,.999))

    
    def fit(self ,batch_Input , batch_Output , maxAge , maxErro,n = 0.05 ,Betas = (0.9,.999) ,  lossFunction = nn.CrossEntropyLoss() , 
            lossGraphNumber = 1 ):
        self.optimizer = torch.optim.Adam(self.parameters(), n ,Betas)
        lossValue = float("inf")
        Age = 0
        lossList = []

        # batch_Input  = [ self.Embedding.sequence2vectors(i) for i in batch_Input  ]
        # batch_Input  = [ self.Embedding.sequence2vectors(i) for i in batch_Input  ]
        # batch_Output = [ self.Embedding.sequence2idx(i)     for i in batch_Output ]
        while lossValue > maxErro and Age < maxAge :
            lossValue = 0
            
            for x,y in zip(batch_Input,batch_Output) :
                print("y.shape[0] = {}".format(y.shape[0]))
                if type(y) != type(torch.tensor([1])) :
                    x = torch.from_numpy(x).float()
                    y = torch.from_numpy(y).float()
                div = len(y)
                enc = self.encoder(x , mask = False ,scale = True )
                print('____________DECODER ___________________\n\n\n\n')
                out = self.decoder.forward_fit(enc , enc , max_lengh = y.shape[0])
                
                # print("out.shape = " , out.shape ,"\nmult_oneHotEncode(self.model_dim, y ).shape = " , mult_oneHotEncode(len(self.Embedding.vocab), y ).shape )
                # loss = lossFunction(out , mult_oneHotEncode(len(self.Embedding.vocab), y ))
                loss = lossFunction(out , y )
                lossValue += loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            Age += 1
            lossValue = lossValue/div
            lossList.append(lossValue)
        plt.plot(range(1 , Age + 1) , lossList)
        if lossGraphNumber != 1 :
            plt.savefig("{}_Tener_LossInTrain_Plot.png".format(lossGraphNumber) )
            plt.savefig("{}_Tener_LossInTrain_Plot.pdf".format(lossGraphNumber) )
        else :
            plt.savefig("Tener_LossInTrain_Plot.png")
            plt.savefig("Tener_LossInTrain_Plot.pdf")
        
        print("O erro final foi de {} ".format(lossValue))

    def forward(self , x ,Enc_mask = False ,Enc_scale = True ,max_lengh = 100 ) :
        enc = self.encoder(x , mask = Enc_mask ,scale = Enc_scale )
        out = self.decoder(enc , enc , max_lengh = max_lengh )
        return out

# embed = skip_Gram()


print("Run executada com sucesso ")