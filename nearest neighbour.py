import math
import numpy as np
import pandas as pd
class city:
    all_points = []
    visited =[]
    unvisited =[]
    def __init__(self,n,x,y,index):
        self.name = n
        self.x = x
        self.y = y
        self.index = index
        city.all_points.append (self)
        city.unvisited.append (self)


    def distance_calc (self , p) :
        dist = math.sqrt(((self.x - p.x)**2)+((self.y -p.y)**2))
        return dist

    @classmethod
    def random_generator (cls):
        return cls.all_points[0]

    def nearest_calc(self, matrix):
        value = np.inf
        ind = np.inf
        for point in city.unvisited:
            if (matrix[self.index][point.index] < value) & (point.index != self.index):
                        ind = point.index
                        value = matrix[self.index][point.index]
        return ind

    @classmethod
    def visited_points (cls, p):
        cls.visited.append(p)
        cls.unvisited=[]
        for x in cls.all_points :
            if x in cls.visited:
                pass
            else:
                cls.unvisited.append(x)
        return cls.visited

    @classmethod
    def distance_mat(cls):
        distances = np.zeros((len(city.all_points), len(city.all_points)))
        for p1 in city.all_points:
            for p2 in city.all_points:
                dist = p1.distance_calc (p2)
                distances[p1.index ,p2.index] = dist
        return distances
    def __repr__(self):
        return f"{self.name}"



df = pd.read_csv ("15-Points.csv")
p = []

for i in range(len(df.City)) :
    p.append( city(df.City[i],df.x[i],df.y[i],i))

matrix = city.distance_mat()
str_point = city.random_generator()
my_point = str_point
cost_total = 0
visited =[p[0]]
while len(city.visited) != len(city.all_points):
    ind = my_point.nearest_calc(matrix)
    my_point = p[ind]
    cost = matrix[visited[-1].index][ind]
    visited = city.visited_points(my_point)
    cost_total = cost_total + cost

print (f"total cost = {cost_total}")
print(visited)







