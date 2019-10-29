from utils import *
import os, shutil

def commons_f(G_s):
    commons=[]

    for m,n in [[0,2],[0,3],[1,2],[1,3]]:
        t=[]
        for i,j in zip(G_s[m].edges,G_s[n].edges):
            if i == j:
                t.append(i)
            else:
                break
        t_=[]
        for i,j in zip(list(G_s[m].edges)[::-1],list(G_s[n].edges)[::-1]):
            if i == j:
                t_.append(i)
            else:
                break
        if len(t) > len(t_):
            commons.append(t)
        else:
            commons.append(t_)
    return commons

class AG:
    
    
    def __init__(self, nombre,persons,
                 individuals,mode_initial_pop,
                 n_generations,
                 p_crossover=0.7,
                 p_mutation=None,
                 crossover_func=ruleta,
                 mutation_func=mutation_usual,
                 generacional=False,
                 fitness_func=fitness_basic):
        
        self.nombre=nombre
        self.persons = persons
        self.gens = int(persons*(persons-1)/2)
        if individuals % 2 == 1:
            individuals+=1
        self.individuals=individuals
        self.current_population=None
        self.mode_initial_pop=mode_initial_pop
        self.fitness_func=fitness_func
        self.fitness={}
        self.n_generations=n_generations
        self.current_generation=1
        self.generations={1:{}}
        self.p_crossover=p_crossover
        self.mutation_func=mutation_func
        if not p_mutation:
            p_mutation=1.0/self.gens
        self.p_mutation=p_mutation
        self.crossover_func=crossover_func
        self.generacional=generacional
        
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        
    def info(self):
        
        I=pd.DataFrame([self.nombre,self.persons,self.gens,
                        self.individuals,self.n_generations,
                        self.p_crossover,self.p_mutation,
                        self.generacional
                       ],
                       index=["Name","Persons","Gens",
                              "individuals","n_generations",
                              "p_crossover","p_mutation",
                              "generacional"
                                
                               ],
                       columns=["Values"]
        )
        return I
        
        
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
            
        plt.show()
            
    def evaluate_fitness_ind(self,individual):

        t_f=t_form(individual)
        
        if t_f not in self.fitness:
            #print(self.fitness_func(individual))
            return self.fitness_func(individual)            
        else:
            #print(self.fitness[t_f])
            return self.fitness[t_f]
            
    def evaluate_pop(self,P,offspring=None):
        values=list(map(self.evaluate_fitness_ind,P))
        #print(values)            
        for i,v in zip(P,values):
            self.fitness[t_form(i)]=v
        
        if not offspring:
            self.generations[self.current_generation]={"Average Fitness":np.mean(values),
                                                       #"Best Individual":P[np.argmax(values)],
                                                       "Fitness Best":np.max(values),
                                                       "Fitness Worst":np.min(values),
            }
            
            return np.array(values),self.generations[self.current_generation] 
        else:
            return values
        
    def get_parents(self):
        
        #print(self.evaluate_pop(self.current_population)[0])
        P=self.crossover_func(self.current_population,self.evaluate_pop(self.current_population)[0])
        return P[::2], P[1::2]
    
    
    

    def crossover(self,P1,P2):
        
        cruce=list(map(crossover_usual,P1,P2,[self.persons]*len(P1),[self.p_crossover]*len(P1)))
        
        return cruce

    def mutation(self,p1,p2,c):
        ind_aux=np.array([p1,p2,c[0],c[1]])
        no_cruce = (t_form(p1) == t_form(c[0])) | (t_form(p1) == t_form(c[1])) | (t_form(p2) == t_form(c[0])) | (t_form(p2) == t_form(c[1]))
        if no_cruce == True:
            return np.array([p1,p2])
        else:
            hijos_mut=np.array(list(map(self.mutation_func,c,[self.persons]*len(c),[self.p_mutation]*len(c))))
            return hijos_mut
        
    def select_best(self,p1,p2,c):
        ind_aux=np.array([p1,p2,c[0],c[1]])
        no_cruce = (t_form(p1) == t_form(c[0])) | (t_form(p1) == t_form(c[1])) | (t_form(p2) == t_form(c[0])) | (t_form(p2) == t_form(c[1]))
        if no_cruce == True:
            return np.array([p1,p2])
        else:
            fitness=list(map(self.evaluate_fitness_ind,ind_aux))
            hijos=ind_aux[np.argpartition(fitness,1)[::-1][:2]]
            return hijos

    
    def evolution(self):
        P1,P2=self.get_parents()
        cruce=self.crossover(P1,P2)
        cruce_mut=list(map(self.mutation,P1,P2,cruce))
        
        if self.generacional == True:
            self.current_population =np.concatenate(cruce_mut)
        else:
            self.current_population = np.concatenate(list(map(self.select_best,P1,P2,cruce_mut)))
                           
        self.current_generation+=1
        self.fitness={}
        #return self.evaluate_pop(self.current_population)[1]   
        return 0
    
    def plot_results(self,size=(12,5),save=None):
        fig = plt.figure()
        fig.set_size_inches(size[0], size[1])
        ax1 = fig.add_subplot(111)
        ax1.set_title("Results")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        
        sns.lineplot(np.arange(self.current_generation-1),
                     [ i['Average Fitness'] for i in self.generations.values()],
                     ax=ax1,ci='sd',
                     label='Average Fitness')
        sns.lineplot(np.arange(self.current_generation-1),
                     [ i['Fitness Best'] for i in self.generations.values()],
                     ax=ax1,
                     label='Fitness Best')
        sns.lineplot(np.arange(self.current_generation-1),
                     [ i["Fitness Worst"] for i in self.generations.values()],
                     ax=ax1,
                     label="Fitness Worst")      
        if save:
            plt.savefig(save,dpi=300)
            
        plt.show()
            
    def run(self,size=(12,5),save_dist=None,save_results=None):
        self.generate_pop_ini()
        self.show_dist_deggre_pop(size=size,save=save_dist)
        for i in range(self.n_generations):
            self.evolution()
            self.show_dist_deggre_pop()
        self.plot_results(size=size,save=save_results)
        
    def run_alone(self,size=(12,5),path_dir="",console=True,plots=10):
        
        if console == True:
            matplotlib.use('Agg')
        
        if os.path.exists(path_dir):
            shutil.rmtree(path_dir)
        os.mkdir(path_dir)
        self.info().to_csv(os.path.join(path_dir,"Info.csv"))
        self.generate_pop_ini()
        #print(self.fitness_func)
        #print(False in [validate_individual(self.persons,i) for i in self.current_population])
        self.show_dist_deggre_pop(size=size,save=os.path.join(path_dir,"Dist_ini.pdf"))
        
        folder_generations=os.path.join(path_dir,"generations")
        os.mkdir(folder_generations)
        for i in range(self.n_generations):
            self.evolution()
            #print(False in [validate_individual(self.persons,i) for i in self.current_population])
            if i in np.around(np.linspace(plots,self.n_generations,plots)) :
                self.plot_results(size=size,save=os.path.join(path_dir,"Results.pdf"))
                self.show_dist_deggre_pop(size=size,save=os.path.join(folder_generations,"generation_"+str(i)+".pdf"))
        
        pd.DataFrame(self.generations).T.to_csv(os.path.join(path_dir,"results_generations.csv"))
        self.plot_results(size=size,save=os.path.join(path_dir,"Results.pdf"))
        values=list(map(self.evaluate_fitness_ind,self.current_population))
        np.save(os.path.join(path_dir,"best.npy"),self.current_population[np.argmax(values)])
        
        return 0
        
    def example_mutation_usual(self,save=None):
        P=self.current_population
        individual=P[np.random.choice(len(P))]
        colors=['black','r']
        fig,ax=plt.subplots(1,3,figsize=(15,5))
        [axi.set_axis_off() for axi in ax.ravel()]
        ax[0].title.set_text('Original')
        ax[1].title.set_text('Mutation')
        ax[2].title.set_text('Highlight Difference')
        G=convert_individual_to_graph(self.persons,individual)
        nx.draw_networkx(G,pos=nx.circular_layout(G),with_labels=True,ax=ax[0],width=2)
        mut=mutation_usual(individual,self.persons)
        X=convert_individual_to_graph(self.persons,mut)
        nx.draw_networkx(X,pos=nx.circular_layout(X),with_labels=True,ax=ax[1],width=2)
        effect=list(set(G.edges).symmetric_difference(set(X.edges)))
        X.add_edges_from(effect)
        nx.draw(X,pos=nx.circular_layout(X),with_labels=True,
                         edge_color=tuple(colors[(i in effect)*1] for i in X.edges),
                         width=2
                        )
        if save:
            plt.savefig(save,dpi=300) 
        plt.show()
        
    def example_crossover_usual(self,save=None):
        
        P=self.current_population
        I=P[np.random.choice(len(P),2)]
        J=crossover_usual(I[0],I[1],self.persons)
        I=np.concatenate((I,J))
        fig,ax=plt.subplots(2,2,figsize=(10,10))
        [axi.set_axis_off() for axi in ax.ravel()]
        ax=ax.ravel()
        [axi.title.set_text(title) for axi,title in zip(ax,['Parent 1','Parent 2','offspring 1','offspring 2'])]
        G_s=[]
        for axi,individual in zip(ax,I):
            G=convert_individual_to_graph(self.persons,individual)
            G_s.append(G)

        colors=np.array(['g','r','b','orange'])
        com=commons_f(G_s)
        c=[]
        c.append([colors[np.where([i in com[0],i in com[1]])[0][0]]  for i in G_s[0].edges])
        c.append([colors[np.where([i in com[2],i in com[3]])[0][0]+2]  for i in G_s[1].edges])
        c.append(colors[::2][[np.where([i in com[0],i in com[2]])[0][0] for i in G_s[2].edges]])
        c.append(colors[1::2][[np.where([i in com[1],i in com[3]])[0][0] for i in G_s[3].edges]])


        for axi,individual,col in zip(ax,G_s,c):
            nx.draw_networkx(individual,pos=nx.circular_layout(individual),with_labels=True,ax=axi,width=2,edge_color=col)
        if save:
            plt.savefig(save,dpi=300) 
        plt.show()
            
            
    def example_swap(self,save=None):
        P=self.current_population
        individual=P[np.random.choice(len(P))]
        colors=['b','r']
        fig,ax=plt.subplots(1,2,figsize=(12,5))
        [axi.set_axis_off() for axi in ax.ravel()]
        ax[0].title.set_text('Original')
        ax[1].title.set_text('Swap')
        swap,a,b=swap_mutation(individual,self.persons,test=True)
        G=convert_individual_to_graph(self.persons,individual)
        nx.draw_networkx(G,pos=nx.circular_layout(G),
                         with_labels=True,ax=ax[0],width=2,node_color=[colors[i in [a,b]] for i in G.nodes])
        G=convert_individual_to_graph(self.persons,swap)
        nx.draw_networkx(G,pos=nx.circular_layout(G),
                         with_labels=True,ax=ax[1],width=2,node_color=[colors[i in [a,b]] for i in G.nodes])
        if save:
            plt.savefig(save,dpi=300) 
        plt.show()

    
        
        
        