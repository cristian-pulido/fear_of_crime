from AG_basic import *
from utils import *
import multiprocessing
from joblib import Parallel, delayed

num_cores = 2#round(multiprocessing.cpu_count()*1/2)

def multi_runs_AG(root_path,runs,nombre,persons,individuals,
                  mode_initial_pop,n_generations,p_crossover,
                  fitness_func,plots,p_mutation=None,
                  crossover_func=ruleta,generacional=False,console=True):
    
    def results(partial_runs):

        results=[]
        for i in range(partial_runs):
          path_run=os.path.join(runs_path,"run_"+str(i))
          results.append(pd.read_csv(os.path.join(path_run,"results_generations.csv"))[['Fitness Best']])
        A=pd.concat(results).reset_index().rename(columns={'index':'Generation'})
        A.to_csv(os.path.join(root_path,"table_results.csv"),index=False)

        fig = plt.figure()
        fig.set_size_inches(10, 5)
        ax1 = fig.add_subplot(111)


        sns.lineplot(data=A,x='Generation',y='Fitness Best',estimator=np.min,ci=None,label='The Best',ax=ax1)
        sns.lineplot(data=A,x='Generation',y='Fitness Best',estimator=np.mean,ci='sd',label='Mean Bests',ax=ax1)
        sns.lineplot(data=A,x='Generation',y='Fitness Best',estimator=np.max,ci=None,label='The Worst',ax=ax1)
        ax1.set_ylabel("Fitness")
        plt.savefig(os.path.join(root_path,"results_multi_run.pdf"))
        plt.show()
    

    def single_AG(i):
        
        AG_1=AG(nombre=nombre+str(i),
              persons=persons,
              individuals=individuals,
              mode_initial_pop=mode_initial_pop,
              n_generations=n_generations,
              p_crossover=p_crossover,
              p_mutation=p_mutation,
              crossover_func=crossover_func,
              generacional=generacional,
              fitness_func=fitness_func)


        AG_1.run_alone(path_dir=os.path.join(runs_path,"run_"+str(i)),console=console,plots=plots)
        if i % 2 == 1:
            try:
                results(i)
            except:
                print("No results")

        return 0




    if console == True:
        matplotlib.use('Agg')


    #if os.path.exists(root_path):
    #    shutil.rmtree(root_path)
    #os.mkdir(root_path)

    runs_path=os.path.join(root_path,"runs")
    os.mkdir(runs_path)

    I=pd.DataFrame([nombre,persons,individuals,n_generations,
                            p_crossover,p_mutation,
                            generacional,runs],
                   index=["Name","Persons",
                          "individuals","n_generations",
                          "p_crossover","p_mutation",
                          "generacional","runs"

                           ],
                   columns=["Values"]
            )
    I.to_csv(os.path.join(root_path,"Info.csv"))

    Parallel(n_jobs=num_cores)(delayed(single_AG)(i) for i in range(runs))

#     for i in range(runs):
#         AG_1=AG(nombre=nombre+str(i),
#                     persons=persons,
#                     individuals=individuals,
#                     mode_initial_pop=mode_initial_pop,
#                     n_generations=n_generations,
#                     p_crossover=p_crossover,
#                     p_mutation=p_mutation,
#                     crossover_func=crossover_func,
#                     generacional=generacional,
#                     fitness_func=fitness_func)


#         AG_1.run_alone(path_dir=os.path.join(runs_path,"run_"+str(i)),console=console,plots=plots)

    results(runs)
   
