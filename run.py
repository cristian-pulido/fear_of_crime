import os, shutil
import sys
# os.chdir("Algoritmos_geneticos/")
# from multi import *
# from utils import *
# os.chdir("../Model")
# from pos import *
# os.chdir("../")

sys.path.append(os.path.join(os.path.dirname(__file__),"Model"))
sys.path.append(os.path.join(os.path.dirname(__file__),"Algoritmos_geneticos"))


from multi import *
from utils import *
from pos import *

def run_experiments(grupos = {'A':'Immune','B':'Susceptible','C':'Highly Susceptible'},
                    modelo='g_m_v',
                    crimen = 3,
                    n=300,
                    T=200,
                    psi=0.98,
                    nu=0.8,
                    mu=0.14,
                    lamda={'A':0,'B':0.005,'C':0.05},
                    q={'A':0.7,'B':0.2,'C':0.1},
                    n_vecinos=10,
                    range_vecinos=5,
                    ## parametros AG,
                    root_path="main_test",
                    runs=30,
                    nombre="fear",
                    individuals=500,
                    mode_initial_pop=pop_ini_uniform,
                    n_generations=50,
                    p_crossover=0.7,
                    p_mutation=None,
                    crossover_func=ruleta,
                    generacional=False,
                    console=True,
                    plots=5
                   ):
    
    
    persons=n
    dist_cr=dist_crimen(crimen,n,np.array(list((q.values()))))
    s0=np.random.rand(n)
    def convert_individual_to_vecinos(individual,dist_crimen,n=n):
        
        vecinos=[[i,c] for i,c in zip(range(n),dist_crimen)]
        #print(vecinos)
        k=0
        for i in range(n-1):    
            I=i*n-k
            F=(i+1)*(n-1)-k
            k+=(i+1)
            #print(individual[I:F])
            v=np.where(individual[I:F])[0]+i+1
            
            for j in v:
                vecinos[i].append(j)
                vecinos[j].append(i)
        return vecinos

    def normpdf(x, mean, sd):
        import math
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom

    def fitness_fear(individual,n_vecinos=n_vecinos,range_vecinos=range_vecinos):
        
        #from Model.pos import generate
        vertices=convert_individual_to_vecinos(individual,dist_cr,n)

        g_mean=np.mean([len(v[2:]) for v in vertices])

        #x=normpdf(g_mean,n_vecinos,2)/normpdf(10,n_vecinos,2)
        x=1
        if g_mean < n_vecinos-range_vecinos or g_mean > n_vecinos+range_vecinos :
                        
            return x
        
        else:

            S=generate(vertices,psi=psi,
                       nu=nu,mu=mu,T=T,
                       s=s0,lamda=lamda,modelo=modelo)[0]
            return S.T[:,100:].mean()*x
    
    fitness_func=fitness_fear
    
    
    
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.mkdir(root_path)
    
    print("Inicio")
    pd.DataFrame({'Vars':['grupos','modelo','crimen','n','T','psi','nu','mu','lamda','q','n_vecinos',
                      'root_path','runs','nombre','individuals','n_generations','p_crossover',
                      'p_mutation','generacional'],
              'Values':[grupos,modelo,crimen,n,T,psi,nu,mu,lamda,q,n_vecinos,
                       root_path,runs,nombre,individuals,n_generations,p_crossover,
                       p_mutation,generacional]
             }).to_csv(os.path.join(root_path,"Parameters.csv"),index=False)

    multi_runs_AG(root_path,runs,nombre,persons,individuals,
                  mode_initial_pop,n_generations,p_crossover,
                  fitness_func,plots,p_mutation,
                  crossover_func,generacional,console)

    print("Fin")

