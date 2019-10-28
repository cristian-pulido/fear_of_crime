import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing

#### utils 
def convert_matrix_to_individual(M):
    n=len(M)
    return np.concatenate([M[i][i+1:n] for i in range(n)])
def convert_individual_to_matrix(n,individual):
    k=0
    for i in range(n-1):    
        I=i*n-k
        F=(i+1)*(n-1)-k
        k+=(i+1)
        M[i][i+1:n]=individual[I:F]
    return M.T+M

def convert_individual_to_graph(n,individual):
   #n=persons
    X=nx.Graph()
    X.add_nodes_from(np.arange(n))
    k=0
    for i in range(n-1):    
        I=i*n-k
        F=(i+1)*(n-1)-k
        k+=(i+1)
        index=np.where(individual[I:F])[0]+(i+1)
        X.add_edges_from(list(zip(i*np.ones(len(index),dtype=int),index)))
    return X

#def convert_graph_to_individual(G):
#    n=G.number_of_nodes()
#    X=nx.adj_matrix(G,np.arange(n))
#    x,y=X.nonzero()
#    x,y=x[x<y],y[x<y]
#    ind=np.zeros(int(n*(n-1)/2))
#    resta=0
#    for i in range(n):
#        j=y[np.where(x == i)[0]]  
#        resta+=i+1
#        ind[i*n+j-resta]=1
#        
#    return ind.astype(int)


def validate_individual(n,individual):
    #n=persons
    X=nx.Graph()
    X.add_nodes_from(np.arange(n))
    k=0
    for i in range(n-1):    
        I=i*n-k
        F=(i+1)*(n-1)-k
        k+=(i+1)
        index=np.where(individual[I:F])[0]+(i+1)
        X.add_edges_from(list(zip(i*np.ones(len(index),dtype=int),index)))
    return nx.is_connected(X)


def t_form(individual):
        return tuple(individual.astype(int))



### operators 

def swap_mutation(individual,n,a=None,b=None,test=False):
    G=convert_individual_to_graph(n,individual)
    if not a:
        a=np.random.choice(G.nodes)
    if not b:
        b=np.random.choice(G.nodes)
    
    A=nx.relabel.relabel_nodes(G,{a:b,b:a})
    
    if test == False:
        return convert_matrix_to_individual(np.asarray(nx.to_numpy_matrix(A,np.arange(n),dtype=int)))
    else:
        return convert_matrix_to_individual(np.asarray(nx.to_numpy_matrix(A,np.arange(n),dtype=int))),a,b

def mutation_usual(individual,n,p=None):
    if not p:
        p=1/len(individual)
    mut = np.random.rand(len(individual))<p    
    result=(individual + mut)%2  
    while validate_individual(n,result) == False:
        mut = np.random.rand(len(individual))<p
        result=(individual + mut)%2 
        
    return result 

def crossover_usual(individual1,individual2,n,p=0.7):
    gens=len(individual1)
    linea_cruce=np.random.randint(0,gens)
    cruce=(np.arange(gens)<linea_cruce)*(np.random.rand()<p)
    h1=individual1*cruce + individual2*(~cruce)
    h2=individual1*~cruce + individual2*cruce
    while False in [validate_individual(n,h1),validate_individual(n,h2)]:
        linea_cruce=np.random.randint(0,gens)
        cruce=(np.arange(gens)<linea_cruce)*(np.random.rand()<p)
        h1=individual1*cruce + individual2*(~cruce)
        h2=individual1*~cruce + individual2*cruce

    return h1,h2



## population ini
def pop_ini_bin_fixed(AG):
    gens=AG.gens
    individuals=AG.individuals
    
    M=np.random.randint(2,size=(individuals,gens))
    AG.current_population=M

def pop_ini_uniform(AG):
    gens=AG.gens
    individuals=AG.individuals
    persons=AG.persons
    
    M=((np.random.rand(gens,individuals) < np.random.rand(individuals))*1).T
    unq, ind_unq= np.unique(M, axis=0,return_index=True)
    M=M[np.sort(ind_unq)]
    index=[]
    for i,j in enumerate(M):
        if validate_individual(persons,j) == True:
            index.append(i)
    M=M[index]

    while len(M) < individuals:
        N=((np.random.rand(gens,individuals-len(M)) < np.random.rand(individuals-len(M)))*1).T
        index=[]
        for i,j in enumerate(N):
            if validate_individual(persons,j) == True:
                index.append(i)
        N=N[index]
        M=np.concatenate((M,N))
        unq, ind_unq= np.unique(M, axis=0,return_index=True)
        M=M[np.sort(ind_unq)]
    AG.current_population=M

def pop_ini_bin(AG):
    
    
    def generate_individual(AG,p=None):
        if not p :
            p=np.random.rand()        
        individual=np.ndarray.tolist(np.random.binomial(n=1,p=p,size=AG.gens))        
        k=0        
        while validate_individual(AG.persons,individual) == False:
            individual=np.ndarray.tolist(np.random.binomial(n=1,p=p,size=AG.gens))
            k+=1
            if k > 100:
                k=0
                p=np.random.rand()                                        
        return np.array(individual)
    
    gens=AG.gens
    individuals=AG.individuals
    persons=AG.persons
    
    P=np.zeros((individuals,gens))
    for i in range(individuals):
        ind=generate_individual(AG)
        P[i]=ind

    unq, ind_unq= np.unique(P, axis=0,return_index=True)

    while len(unq) < individuals:
        for i in set(np.arange(individuals))-set(ind_unq):
            ind=generate_individual(AG)
            P[i]=ind
        unq, ind_unq= np.unique(P, axis=0,return_index=True)

    AG.current_population=P
    
    
### fitness functions    

def fitness_basic(individual):
    return individual.sum()


    
#### metodos seleccion padres
def permutation(P,F=None):
    return np.random.permutation(P)

def random(P,F=None):
    return P[np.random.choice(len(P),size=len(P))]
    
def ruleta(P,F):
    F=F/F.sum()
    return P[np.random.choice(len(P),p=F,size=len(P))]
