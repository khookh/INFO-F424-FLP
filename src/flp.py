import pyomo.environ as pyo
import sys
import time


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


# cost to open each factory, demand of each customer, capacity of each factory, travel cost to customer i from factory j


def solve_flp(instance_name, linear):
    instance_param = read_instance(instance_name)
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, len(instance_param[1]) - 1)
    model.J = pyo.RangeSet(0, len(instance_param[2]) - 1)
    model.f = pyo.Param(model.J, initialize=instance_name[0], default=0)
    model.c = pyo.Param(model.J, initialize=instance_name[1], default=0)
    model.d = pyo.Param(model.I, initialize=instance_name[2], default=0)
    model.t = pyo.Param(model.I, model.J, initialize=instance_name[3], default=0)

    model.x = pyo.Var(model.I, model.J)  # integer amount of client i that is satisfied by facility j
    model.y = pyo.Var(model.J)  # yj = 1 if factory j is built

    def obj_rule():
        return (pyo.summation(model.y, model.f)  # cost of opening all facilities
                + pyo.summation(model.x, model.t)  # cost of moving units from facilities to clients
                )

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    opt = pyo.SolverFactory('glpk')
    opt.solve(model, tee=True)
    print(pyo.value(model.obj))
    # return (obj,x,y)


def initial_solution_flp(instance_name):
    pass
    # return (obj,x,y)


def local_search_flp(x, y):
    pass
    # return (obj,x,y)


if __name__ == '__main__':
    solve_flp('FLP-100-20-1.txt', False)  # test
