# -*- coding: utf-8 -*-
"""
@author: Alexander Nina     5950236
"""

import numpy as np
import random
import array
from deap import algorithms
from deap import creator
from deap import base
from deap import tools

# matriz cargada de csv
diskm=np.genfromtxt('C:/Users/aaale/Desktop/TSPDatos.csv', delimiter=",")       
ciudades=[x for x in range(len(diskm[0]))]
print(diskm)


#para minimizar la distancia min
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array,typecode='i',fitness=creator.FitnessMin)
toolbox=base.Toolbox()

#aquí se usa el random.sample de 0 a 8 sin repetidos
toolbox.register("indices", random.sample, range(len(ciudades)),len(ciudades))

#100 of indices elementos
toolbox.register("individual",tools.initIterate,creator.Individual, toolbox.indices)
#la poblacion de individual 
toolbox.register("population", tools.initRepeat,list, toolbox.individual)

# para sumar las distancias de la matriz creada fitness
def sumaEval(individual):
    sumatoria=0
    inicio=individual[0]
    for i in range(1, len(individual)):
        fin=individual[i]
        sumatoria+=diskm[inicio][fin]
        inicio=fin
    sumatoria+=diskm[fin][individual[0]]
    return sumatoria,

# se usa el cxpartialymatched y mutshuffleindexes del Nqueens
toolbox.register("evaluate", sumaEval)
#cruce en 2 puntos mas cruces
toolbox.register("mate", tools.cxPartialyMatched)

#lo muta con probabilidad de 0.01 diversificación genética de la pob inv. de genes - (cambio de orden)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1/len(ciudades))

#se elije al mejor individuo que tenga mayor puntuación para reproducirlo no compara toda la pobalción
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
##mismo main de nqueens con otra población
    #no encuentre repeticion
    random.seed(169)
    #200 individuals
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 40, stats=stats,halloffame=hof, verbose=True)
    
    return pop, stats

if __name__ == "__main__":
    pop, stats=main()

        