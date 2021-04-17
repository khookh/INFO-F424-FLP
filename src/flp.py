import pyomo.environ as pyo
import sys


def read_instance(file_name):
    opening_cost = {}
    demand = {}
    capacity = {}
    travel_cost = {}
    try:
        file = open("Instances/{}".format(file_name), 'r')
        info = file.readline().split(" ")
        I = int(info[0])
        J = int(info[1])
        info = file.readline().split(" ")
        for j in range(J):
            opening_cost[j] = int(info[j])
        info = file.readline().split(" ")
        for i in range(I):
            demand[i] = int(info[i])
        info = file.readline().split(" ")
        for j in range(J):
            capacity[j] = int(info[j])
        for i in range(I):
            info = file.readline().split(" ")
            for j in range(J):
                travel_cost[(i, j)] = int(info[j])
    except:
        print("Error reading file.")
    return opening_cost, demand, capacity, travel_cost
# cost to open each factory, demand of each customer, capacity of each factory, travel cost from factory i to customer j


def solve_flp(instance_name, linear):
    print(read_instance(instance_name))  # test
    pass
    # return (obj,x,y)


def initial_solution_flp(instance_name):
    pass
    # return (obj,x,y)


def local_search_flp(x, y):
    pass
    # return (obj,x,y)


if __name__ == '__main__':
    solve_flp('FLP-100-20-1.txt', False)  # test
