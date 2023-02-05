"""Script to perform genetic algorithm optimisation on the spike sorter"""
import ss 
import GA       #self-made GA from CW-B
import pickle

#get training data
d, training_labels = ss.load_training_data()
sd = ss.load_submission_data()

#generate dict of all paramaters and their possible values
wavelets = ['haar', 'db4', 'bior1.5', 'coif2', 'rbio3.3', 'sym4']
spike_sorter_params = {
    'hpFreq': [20, 50, 100, 150, 200],
    'denoisingWavelet': wavelets,
    'denoising_thresh_coeff': [0.4, 0.6, 0.8, 1, 1.2],
    'thK': [3,4,5,6],
    'clipSize': [32, 64, 128],
    'clipDistDef': [[15,100], [20,100], [25,100], [30,100]],
    'dwtWavelet': wavelets,
    'svmKernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    
}

#dict with params that will not change
spike_sorter_data = {
    'data': d,
    'sub_data': sd,
    'Fs': 25e3,
    'labels': training_labels,
    'training_data_ratio': 0.8,
    'evalTol': 10,
    'fp_rejection': True,
    'denoisingLevel': 1,
}

#create GA dict with GA hyper-parameters
ga_params = {
    'pop_count': 20,
    'params': spike_sorter_params,
    'data': spike_sorter_data,
    'survival_ratio': 0.8,
    'random_select': 0.01,
    'elitism_count': 1,
    'mutation_rate': 0.5
}

#create the GA
ga = GA.GA_SpikeSorting(**ga_params)
print(f"Gen {0}, fitness: {ga.best.fitness}, genome {ga.best.genome}")

#evolve for 100 generation maximally
for n in range(1,100):
    ga.evolve()
    print(f"Gen {n}, fitness: {ga.best.fitness}, genome {ga.best.genome}")
    #save latest GA object in case bug/crash
    file = open("GA_results/ss_params", 'wb')
    pickle.dump(ga.best.genome, file)
    file.close()

