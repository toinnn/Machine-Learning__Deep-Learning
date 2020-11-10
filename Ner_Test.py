if __name__ == "__main__" :
    from Tener import Tener
    from skip_Gram import skip_gram
    import os 
    import json
    import torch
    import torch.tensor as tensor
    import pickle
    from gensim.models.keyedvectors import KeyedVectors
    import re
    import numpy as np

    def word2vec(wordVec , word ,dim):
        try:
            
            return torch.from_numpy(wordVec[word]).view(1,-1)
        except :
            print("Entrou no vetor não pertencente ao vocabulario ")
            return torch.ones([1,dim]).float()*-79
    
    def sen2vec(gensimWorVec , sentence , dim) :
        sen = re.split(r"(\W)", sentence )
        if len(sen)>1 :
            sen = [token.lower() for token in sen if token != " " and token != "" ]
        else :
            sen = [token for token in (sen[0].lower()) if token != " " and token != "" ]
        print(sen)
        return torch.cat( [ word2vec(gensimWorVec , word ,dim ) for word in sen ] , dim = 0 )
    
    def json2vec(js , key ,dim , gensimWorVec ):
        try :
            seq = [ vec for sen in js[key] for vec in (sen2vec(gensimWorVec , sen.lower() , dim ) , torch.ones([1,dim])*-127 ) ]
            seq[-1] = torch.ones([1,dim])*23
            return torch.cat( seq , dim = 0 )
        except :
            # Key não existente :
            print("Entrou no não existe a key ")
            return torch.ones([1,dim]).float()*-47
    
    def word2idx(gensimWorVec , word ) :
        try :
            return gensimWorVec.vocab[word].index
        except :
            return gensimWorVec.vocab["<OOV>"].index
    def sen2idx(gensimWorVec , sentence ):
        return [ word2idx( gensimWorVec , token.lower() ) for token in re.split( r"(\W)", sentence ) if token != " " and token != "" ]

    def json2idx(js , key ,gensimWorVec ):
        try :
            return tensor([ index for sen in js[key] for index in sen2idx(gensimWorVec , sen ) + [gensimWorVec.vocab["<Separador>"].index ]][:-1])
        except :
            # Key não existente :
            print("Entrou no não existe a key ")
            return tensor([gensimWorVec.vocab["<key_Vazia>"].index])

    # -47 : Key não existente ; -127 : separador entre targets diferentes ; -79 : fora do vocab ; 23 : End-Of-Sentence
        

    embeddingPath = "EmbeddingBaixados\\Word2Vec_skip_s50\\skip_s50.txt"
    inputPath  ="leNer-Dataset\\raw-Text\\"
    outputPath = "leNer-Dataset\\Json-Ner\\"

    ark_Input  = [inputPath + ark  for ark in os.listdir(inputPath)]
    ark_Output = [outputPath + ark for ark in os.listdir(outputPath)]


    ark = [open( i , "r" , encoding = "utf8").read() for i in ark_Input ]
    ark_Target = [json.load(open(i , "rb")) for i in ark_Output]


    # embed = skip_gram(ark)
    # embed.tokenize()

    vectorDim = 30 
    n         = 0.005
    momentum  = 0.01
    maxAge    = 1
    maxErro   = 0.005


    # print(embed.trainVectors(vectorDim ,n,momentum, maxAge , maxErro ,saveNewPairs = True , device = torch.device("cpu") ) )

    basePath = "C:\\Users\\limaa\\PythonProjects\\VsCodePython\\KPMG\\"
    saveSkipGramFile = basePath + "1-n={}_dim={}_maxAge={}_momentum={}_maxErro={}.savedSkipGram".format(
        n,vectorDim,maxAge , momentum ,maxErro)
    # svFile = open(saveSkipGramFile , "wb")
    # pickle.dump(embed , svFile)

    # loadFile = open(saveSkipGramFile , "rb")
    # embed = pickle.load(loadFile)

    # print(embed.vocabulary)
    wv = KeyedVectors.load_word2vec_format(embeddingPath , binary = False)
    # print(wv["direito"])
    # print(sen2vec(wv,"funciona mesmo isso," , 50))

    # -47 : Key não existente ; -127 : separador entre targets diferentes ; -79 : fora do vocab ; 23 : End-Of-Sentence
    wv["<OOV>"] = np.ones([1,50])*-79
    wv["<Separador>"] = np.ones([1,50])*-127
    wv["<key_Vazia>"] = np.ones([1,50])*-47
    wv["<EOS>"] = np.ones([1,50])*23

    print(ark[0] ,"\n" , ark_Target[0]["materias"]  )

    ark        = [ sen2vec(wv , i , 50 ) for i in ark ]
    ark_Target = [ json2idx(js , "materias"  , wv ) for js in ark_Target ]

    print(ark[0] ,"\n" , ark_Target[0]  )
    # wv["<EOS>"] = np.ones(50)*-79
    
    # print(wv["<EOS>"])
    print(len(wv.vocab))
    model = Tener(50 , 5 ,5 , 6 , 6 ,wv , np.ones([1,50])*23 ) 
    model.fit(ark , ark_Target , 10 , 0.005 , n = 0.05 )
    print("Final")
    print(wv.vocab["<key_Vazia>"].index)

    