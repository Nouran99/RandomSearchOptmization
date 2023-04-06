import random
import math
import numpy as np
import pandas as pd

class city:
    all_points = []
    def __init__(self,n,x,y,index):
        self.name = n
        self.x = x
        self.y = y
        self.index = index
        city.all_points.append(self)

    def distance_calc(self, p):
        dist = math.sqrt(((self.x - p.x)**2)+((self.y -p.y)**2))
        return dist

    def __repr__(self):
        return f"{self.name}"


class AntColonyOptimizer:
    def __init__(self, num_ants, num_iterations, decay_rate, alpha, beta, cities):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta
        self.cities = cities
        self.distances = np.zeros((len(cities), len(cities)))
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j:
                    self.distances[i][j] = cities[i].distance_calc(cities[j])
        self.pheromones = np.ones((len(cities), len(cities))) * 0.1
        self.best_path = None
        self.best_path_length = float("inf")

    def optimize(self):
        for i in range(self.num_iterations):
            ant_paths = []
            for j in range(self.num_ants):
                path = self.generate_ant_path()
                ant_paths.append(path)
                length = self.calculate_path_length(path)
                if length < self.best_path_length:
                    self.best_path = path
                    self.best_path_length = length
            self.update_pheromones(ant_paths)
            self.pheromones *= self.decay_rate

    def generate_ant_path(self):
        path = [random.randint(0, len(self.cities) - 1)]
        while len(path) < len(self.cities):
            current_city = path[-1]
            unvisited_cities = list(set(range(len(self.cities))) - set(path))
            probabilities = []
            for city in unvisited_cities:
                probabilities.append(self.calculate_probability(current_city, city))
            probabilities = np.array(probabilities) / sum(probabilities)
            next_city = np.random.choice(unvisited_cities, p=probabilities)
            path.append(next_city)
        return path

    def calculate_probability(self, current_city, next_city):
        pheromone = self.pheromones[current_city][next_city]
        distance = self.distances[current_city][next_city]
        numerator = (pheromone ** self.alpha) * ((1 / distance) ** self.beta)
        denominator = sum(
            [(self.pheromones[current_city][j] ** self.alpha) * ((1 / self.distances[current_city][j]) ** self.beta) for
             j in range(len(self.cities)) if j != current_city])
        probability = numerator / denominator
        return probability

    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.distances[path[i]][path[i + 1]]
        length += self.distances[path[-1]][path[0]]
        return length

    def update_pheromones(self, ant_paths):
        for i in range(len(self.cities)):
            for j in range(len(self.cities)):
                if i != j:
                    delta_pheromone = 0
                    for path in ant_paths:
                        if j in path and path.index(j) == path.index(i) + 1:
                            delta_pheromone += 1 / self.calculate_path_length(path)
                    self.pheromones[i][j] = (1 - self.decay_rate) * self.pheromones[i][j] + delta_pheromone


df = pd.read_csv ("15-Points.csv")
cities = []

for i in range(len(df.City)) :
    cities.append(city(df.City[i],df.x[i],df.y[i],i))


# set the hyperparameters
num_ants = 20
num_iterations = 200
decay_rate = 0.5
alpha = 1
beta = 2

optimizer = AntColonyOptimizer(num_ants, num_iterations, decay_rate, alpha, beta, cities)
optimizer.optimize()

print("Best Path: ", optimizer.best_path)
print("Best Path Length: ", optimizer.best_path_length)
