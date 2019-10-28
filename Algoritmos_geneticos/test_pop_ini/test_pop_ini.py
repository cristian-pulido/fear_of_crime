import os
import sys

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#print(sys.path)
from utils import *

class AG_pop_ini_test:
    
    def __init__(self, nombre,persons,
                 individuals,mode_initial_pop):
        self.nombre=nombre
        self.persons = persons
        self.gens = int(persons*(persons-1)/2)
        if individuals % 2 == 1:
            individuals+=1
        self.individuals=individuals
        self.mode_initial_pop=mode_initial_pop
                
    def generate_pop_ini(self):
        self.mode_initial_pop(self)
        
    def validate_grade_individual(self,individual):
        n=self.persons
        X=convert_individual_to_graph(n,individual)
        return sum(dict(nx.degree(X)).values())/n
        
    def show_dist_deggre_pop(self,size=(12,5),save=None):
        D=list(map(self.validate_grade_individual,self.current_population))
        fig = plt.figure()
        fig.set_size_inches(size[0], size[1])
        ax1 = fig.add_subplot(121)
        ax1.set_title("Distribution")
        ax2 = fig.add_subplot(122)
        ax2.set_title("Points")
        ax1.set_xlabel("Average Degree")
        ax1.set_ylabel("Frequency")
        ax2.set_xlabel("Average Degree")
        sns.distplot(D,kde=False,ax=ax1)
        sns.swarmplot(x=D,ax=ax2)
        
        if save:
            plt.savefig(save,dpi=300)
            
def test(persons=100,individuals=800,size=(12,5),saves=[None]*3):            
    for idx,i in enumerate([pop_ini_uniform,pop_ini_bin,pop_ini_bin_fixed]):
        AG_1=AG_pop_ini_test(nombre="ini",
                             persons=persons,
                             individuals=individuals,
                             mode_initial_pop=i,  # pop_ini_uniform / pop_ini_bin / pop_ini_bin_fixed
           )
        #%timeit AG_1.generate_pop_ini()
        AG_1.generate_pop_ini()
        AG_1.show_dist_deggre_pop(size=size,save=saves[idx])
    