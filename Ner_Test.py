if __name__ == "__main__" :
    from Tener import Tener
    from skip_Gram import skip_gram
    from BiLSTM import BiLSTM
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

    def jsonList2classes(jsList , key):
        classes = {}
        classifiedList = []
        for js in jsList :
            try :
                if type(js[key])==type([]) :
                    Classified = []
                    for content in js[key] :
                        if content not in classes.keys() :
                            classes[content] = len(classes.keys())
                        Classified += [classes[content]]
                else :
                    if js[key] not in classes.keys() :
                        classes[js[key]] = len(classes.keys())
                    Classified  = [classes[js[key]]]
                classifiedList += [torch.tensor(Classified)]
            except :
                classifiedList += torch.tensor([-1])
        print("Classes igual a " , classes )
        return classes , [ i if (i !=-1).all() else torch.tensor([len(classes.keys())]) for i in classifiedList  ]
    # -47 : Key não existente ; -127 : separador entre targets diferentes ; -79 : fora do vocab ; 23 : End-Of-Sentence
        

    # embeddingPath = "EmbeddingBaixados\\Word2Vec_skip_s50\\skip_s50.txt"
    # inputPath  ="leNer-Dataset\\raw-Text\\"
    outputPath = "leNer-Dataset\\Json-Ner\\"

    # ark_Input  = [inputPath + ark  for ark in os.listdir(inputPath)]
    ark_Output = [outputPath + ark for ark in os.listdir(outputPath)]


    # ark = [open( i , "r" , encoding = "utf8").read() for i in ark_Input ]
    ark_Target = [json.load(open(i , "rb")) for i in ark_Output]

    # materias = []
    # for i in ark_Target :
    #     try :
    #         if type(i["materias"]) == type([]) :
    #             for j in i["materias"] :
    #                 # materias += [j]
    #                 if j not in materias :
    #                     materias += [j]
    #         else :
    #             if i["materias"] not in materias :
    #                 materias += [i["materias"]]
    #     except :
    #         pass
    # materias = ['Direito Civil', 'Direito Processual Civil', 'Direito Administrativo', 'Direito Constitucional',
    #     'Direito Penal', 'Direito Processual do Trabalho', 'Direito do Trabalho', 'Direito Penal Militar',
    #     'Direito Processual Penal', 'Direito Eleitoral', 'Direito Processual Penal Militar', 'Direito Previdenciário', '']
    # print("as materias únicas são " , materias ) #12 classes e uma indicando que não tem
    # print(ark_Target[0]["materias"])
    # teste = "palavra"
    # print(teste)
    # teste = [teste] if type(teste)==type("exe") 
    # print(teste)
    # embed = skip_gram(ark)
    # embed.tokenize()

    vectorDim = 30 
    n         = 0.005
    momentum  = 0.01
    maxAge    = 1
    maxErro   = 0.005


    # print(embed.trainVectors(vectorDim ,n,momentum, maxAge , maxErro ,saveNewPairs = True , device = torch.device("cpu") ) )

    basePath = "C:\\Users\\limaa\\PythonProjects\\VsCodePython\\KPMG\\"
    # saveSkipGramFile = basePath + "1-n={}_dim={}_maxAge={}_momentum={}_maxErro={}.savedSkipGram".format(
    #     n,vectorDim,maxAge , momentum ,maxErro)
    # svFile = open(saveSkipGramFile , "wb")
    # pickle.dump(embed , svFile)

    # loadFile = open(saveSkipGramFile , "rb")
    # embed = pickle.load(loadFile)

    
    # wv = KeyedVectors.load_word2vec_format(embeddingPath , binary = False)
    

    # -47 : Key não existente ; -127 : separador entre targets diferentes ; -79 : fora do vocab ; 23 : End-Of-Sentence
    # wv["<OOV>"] = np.ones([1,50])*-79
    # wv["<Separador>"] = np.ones([1,50])*-127
    # wv["<key_Vazia>"] = np.ones([1,50])*-47
    # wv["<EOS>"] = np.ones([1,50])*23

    # print(ark[0] ,"\n" , ark_Target[0]["materias"]  )
    
    # ark        = [ sen2vec(wv , i , 50 ) for i in ark ]
    # ark_Target = [ json2idx(js , "materias" , wv) for js in ark_Target ]
    # classes , ark_Target = jsonList2classes(ark_Target , "materias")

    # print(ark[0] ,"\n" , ark_Target[0]  )
    # wv["<EOS>"] = np.ones(50)*-79
    
    # pickle.dump(wv , open("wv_W2Vec.pickle","wb"))
    # pickle.dump(ark , open("ark_W2Vec.pickle","wb"))
    # pickle.dump(ark_Target , open("ark_Target_W2Vec.pickle","wb"))
    # pickle.dump(classes , open("classesTarget_Entity_materias_W2Vec.pickle","wb"))

    #O Git-Hub não permite que seja upado um arquivo tão grande quanto wv_W2Vec.pickle , então pode descomentar a linha 110 para gerar
    #o wv serializado e depois comentar as linhas de manipulação do wv e descomentar a linha abaixo para conseguir uma execução + rápida
    wv = pickle.load(open("wv_W2Vec.pickle","rb"))
    ark = pickle.load(open("ark_W2Vec.pickle","rb"))
    ark_Target = pickle.load(open("ark_Target_W2Vec.pickle","rb"))
    classes = pickle.load(open("classesTarget_Entity_materias_W2Vec.pickle","rb"))
    
    # print(wv["<EOS>"])
    print(ark_Target)
    # model = Tener(50 , 5 ,5 , 6 , 6 ,wv , np.ones([1,50])*23 ) 
    # model.fit(ark , ark_Target , 10 , 0.005 , n = 0.05 , lossGraphNumber = 1 )
    # pickle.dump(model , open("1_TenerTreinado_maxAge=10_maxErro=0.005_n=0.05.pickle" , "wb"))
    lstm = BiLSTM(50 ,100 , 1, 100,1 , len(classes.keys()) + 1, wv , torch.ones([1,50])*23 )
    # lstm.fit(ark, ark_Target , 0.05 ,0.06 , 20 )
    # lstm.fit([i.view(1 , i.shape[0] , i.shape[1] ) for i in ark ], ark_Target , 0.05 ,0.06 , 10 )
    # pickle.dump(lstm , open("/content/drive/My Drive/Aprender a Usar A nuvem_Rede-Neural/lstm_n=0.05_maxErro=0.06_maxAge=10.pickle","wb"))
    print("len Ark = {}  , len target = {}".format(len(ark) , len(ark_Target)))

    # print(tuple(i.shape for i in lstm.encoder(ark[0].view(1,ark[0].shape[0] , ark[0].shape[1] )) ) )
    print("Final")
    print(wv.vocab["<key_Vazia>"].index) 

    