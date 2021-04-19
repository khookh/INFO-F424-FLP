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
    return opening_cost, demand, capacity, travel_cost  # cost to open each factory, demand of each customer
    # capacity of each factory, travel cost to customer i from factory j


def solve_flp(instance_name, linear):
    instance_param = read_instance(instance_name)

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, len(instance_param[1]) - 1)
    model.J = pyo.RangeSet(0, len(instance_param[2]) - 1)
    model.f = pyo.Param(model.J, initialize=instance_param[0], default=0)  # factory opening cost (J)
    model.c = pyo.Param(model.J, initialize=instance_param[2], default=0)  # factory capacity (J)
    model.d = pyo.Param(model.I, initialize=instance_param[1], default=0)  # customer demand (I)
    model.t = pyo.Param(model.I, model.J, initialize=instance_param[3], default=0)  # cost to move 1 unit from J to I

    # integer amount of client i demand that is satisfied by facility j
    model.x = pyo.Var(model.I, model.J, domain=pyo.NonNegativeReals if linear else pyo.NonNegativeIntegers)
    # yj = 1 if factory j is built, 0 else
    model.y = pyo.Var(model.J, domain=pyo.PercentFraction if linear else pyo.Binary)

    def obj_rule(_model):
        return (
                pyo.summation(_model.f, _model.y)  # cost of opening facilities
                + pyo.summation(_model.t, _model.x)  # cost of moving units from facilities to clients
        )
        # sum(instance_name[0][j] * model.y[j] for j in model.J) + sum(
        #             instance_name[3][i, j] * model.x[i, j] for i in model.I for j in model.J)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)  # minimize the cost of operation

    def cst1(_model, j):
        return (
                sum(_model.x[i, j] for i in _model.I) <= _model.y[j] * _model.c[j]
        )

    # the sum of the units sent from one factory to all its customer cannot be above its capacity (if it's open)
    model.constraint_1 = pyo.Constraint(model.J, rule=cst1)

    def cst2(_model, i):
        return (
                sum(_model.x[i, j] for j in _model.J) >= _model.d[i]
        )

    model.constraint_2 = pyo.Constraint(model.I, rule=cst2)  # for each customer all the demand must be satisfied

    opt = pyo.SolverFactory('glpk')

    opt.solve(model, tee=True)

    list_x = [[0 for x in range(len(instance_param[2]))] for y in range(len(instance_param[3]))]
    list_y = [0 for x in range(len(instance_param[2]))]

    for i in model.I:
        for j in model.J:
            list_x[i][j] = model.x[i, j].value

    for j in model.y:
        list_y[j] = model.y[j].value

    return pyo.value(model.obj), list_x, list_y


def initial_solution_flp(instance_name):
    pass
    # return (obj,x,y)


def local_search_flp(x, y):
    pass
    # return (obj,x,y)


if __name__ == '__main__':
    print(solve_flp(str(sys.argv[1]), False))  # test
