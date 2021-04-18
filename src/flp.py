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


def test():  # test pyomo concrete model functions
    model = pyo.ConcreteModel()

    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)

    model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])

    model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)


def solve_flp(instance_name, linear):
    instance_param = read_instance(instance_name)
    print(instance_param[0])
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, len(instance_param[1]) - 1)
    model.J = pyo.RangeSet(0, len(instance_param[2]) - 1)
    model.f = pyo.Param(model.J, initialize=instance_param[0], default=0)  # factory opening cost (J)
    model.c = pyo.Param(model.J, initialize=instance_param[2], default=0)  # factory capacity (J)
    model.d = pyo.Param(model.I, initialize=instance_param[1], default=0)  # customer demand (I)
    model.t = pyo.Param(model.I, model.J, initialize=instance_param[3], default=0)  # cost to move 1 unit from J to I

    model.x = pyo.Var(model.I, model.J,
                      domain=pyo.PositiveIntegers)  # integer amount of client i that is satisfied by facility j
    model.y = pyo.Var(model.J, domain=pyo.Binary)  # yj = 1 if factory j is built

    def obj_rule(model):
        return (pyo.summation(model.f, model.y)  # cost of opening all facilities
                + pyo.summation(model.t, model.x)  # cost of moving units from facilities to clients
                )
        # sum(instance_name[0][j] * model.y[j] for j in model.J) + sum(
        #             instance_name[3][i, j] * model.x[i, j] for i in model.I for j in model.J)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    # TODO : ADD CONSTRAINTS
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
    test()
    solve_flp('FLP-100-20-1.txt', False)  # test
