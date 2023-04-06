import math , random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Population_size = 50
Generations_Count = 100
Elitism_percentage = 0.02
Crossover_probability = 0.6
Mutation_probability = 0.1
num_itter = 200

class Chromosome:
    all_chrom =[]
    def __init__(self , lista ,fitness, cost ):
        self.lista =lista
        self.fitness =fitness
        self.cost =cost

    def fitness_calc(self):
        fitness = 1/(Gene.distance_mat()[Gene.index].sum())
        return fitness

    def cost_calc(self):
        distance = 0
        for j in range (len(self.lista)-1):
            distance += self.lista[j].distance_calc(self.lista[j+1])
        distance +=self.lista[0].distance_calc(self.lista[-1])
        return distance

    def __repr__(self):
        return f"{self.lista}"

class Gene:
    all_genes = []
    def __init__(self,n,x,y,index):
        self.name = n
        self.x = x
        self.y = y
        self.index = index
        Gene.all_genes.append (self)

    def distance_calc (self , p) :
        dist = math.sqrt(((self.x - p.x)**2)+((self.y -p.y)**2))
        return dist

    @classmethod
    def random_generator (cls):
        return cls.all_genes[0]

    @classmethod
    def distance_mat(cls):
        distances = np.zeros((len(cls.all_genes), len(cls.all_genes)))
        for p1 in cls.all_genes:
            for p2 in cls.all_genes:
                dist = p1.distance_calc (p2)
                distances[p1.index,p2.index] = dist
        return distances

    def __repr__(self):
        return f"{self.name}"

def selection (population):
    item1,item2,item3,item4 = random.sample(range(len(population)),4)
    cand_1 = population[item1]
    cand_2 = population[item2]
    cand_3 = population[item3]
    cand_4 = population[item4]
    if cand_1.fitness > cand_2.fitness :
        winner1 = cand_1
    else:
        winner1 = cand_2
    if cand_3.fitness > cand_4.fitness :
        winner2 = cand_3
    else:
        winner2 = cand_4
    if winner1.fitness > winner2.fitness :
        winner = winner1
    else:
        winner = winner2

    return winner

def crossover (p1 , p2):
    point = int(Crossover_probability*len(p1))
    child1 = p1.lista[0:point]
    child2 = p2.lista[0:point]
    rest_of_child1 = [item for item in p2.lista if item not in child1 ]
    rest_of_child2 = [item for item in p1.lista if item not in child2 ]

    child1 += rest_of_child1
    child2 += rest_of_child2

    return child1 ,child2

def mutation (chrom):
    item1, item2 = random.sample(range(len(chrom.lesta)), 2)
    chrom.lesta[item1] ,chrom.lesta[item2] =chrom.lesta[item2],chrom.lesta[item1]
    return chrom

def find_elite (population):
    elite = population[0]
    for n in range (len(population)):
        if population[n].fitness > elite.fitness:
            elite =population[n]
    return elite


df = pd.read_csv ("15-Points.csv")
g = []

for i in range(len(df.City)) :
    g.append( Gene(df.City[i],df.x[i],df.y[i],i))

old_population =[]
New_population =[]

def permutation  (list_genes , end = []) :
    lista = []
    for i in range(len(list_genes)):
        lista.append(permutation(list_genes[:i] + list_genes[i + 1:], end + list_genes[i:i + 1]))
    return lista

for i in range(Population_size):
    old_population.append(permutation(Gene.all_genes))
    for pop in old_population:
        r = Chromosome (pop,0,1000)
        r.fitness= r.fitness_calc()
        r.cost = r.cost_calc()
        pop = r

for k in range (num_itter):
    new_population =[find_elite(old_population)]

    for i in range(len(old_population)//2):
        parent1 = selection(old_population)
        parent2 = selection(old_population)
        child1 , child2 = crossover(old_population)

        child1 = Chromosome(child1, 0, 1000)
        child1.fitness = child1.fitness_calc()
        child1.cost = child1.cost_calc()

        child2 = Chromosome(child2, 0, 1000)
        child2.fitness = child2.fitness_calc()
        child2.cost = child2.cost_calc()

        if random.random() < Mutation_probability:
            new_chiled = mutation(child1)

        new_population.append(child1)
        new_population.append(child2)

    old_population =new_population

print(new_population)




