import pyomo.environ as pyo
import numpy as np
import sys
import time
import os


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


def test():  # test pyomo concrete model functions
    model = pyo.ConcreteModel()

    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)

    model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])

    model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)


def solve_flp(instance_name, linear):
    instance_param = read_instance(instance_name)

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, len(instance_param[1]) - 1)
    model.J = pyo.RangeSet(0, len(instance_param[2]) - 1)
    model.f = pyo.Param(model.J, initialize=instance_param[0], default=0)  # factory opening cost (J)
    model.c = pyo.Param(model.J, initialize=instance_param[2], default=0)  # factory capacity (J)
    model.d = pyo.Param(model.I, initialize=instance_param[1], default=0)  # customer demand (I)
    model.t = pyo.Param(model.I, model.J, initialize=instance_param[3], default=0)  # cost to move 1 unit from J to I

    model.x = pyo.Var(model.I, model.J,
                      domain=pyo.NonNegativeIntegers if linear else pyo.NonNegativeReals)  # integer amount of client i demand that is satisfied by facility j
    model.y = pyo.Var(model.J, domain=pyo.Binary if linear else pyo.PercentFraction)  # yj = 1 if factory j is built, 0 else

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

    model.constraint_1 = pyo.Constraint(model.J, rule=cst1)  # the sum of the units sent from one factory to all its

    # customer cannot be above its capacity (if it's open)

    def cst2(_model, i):
        return (
                sum(_model.x[i, j] for j in _model.J) >= _model.d[i]
        )

    model.constraint_2 = pyo.Constraint(model.I, rule=cst2)  # for each customer all the demand must be satisfied

    opt = pyo.SolverFactory('glpk')
    opt.solve(model, tee=True)
    
    return pyo.value(model.obj), model.x, model.y


def initial_solution_flp(instance_name):
    instance_param = read_instance(instance_name)
    customer_nb = len(instance_param[1])
    location_nb = len(instance_param[2])
    fac_opening_cost = instance_param[0]
    customer_demand = instance_param[1]
    fac_capacity = instance_param[2]

    print('There are {} customers.'.format(customer_nb))
    print('There are {} possible factories.'.format(location_nb))

    # Initialize integer solution values
    x = np.zeros((customer_nb, location_nb), dtype=np.int)
    y = np.zeros(location_nb, dtype=np.bool)

    x_opt = np.zeros_like(x, dtype=np.float)
    y_opt = np.zeros_like(y, dtype=np.float)

    # Get relaxed LP solution values
    opt_val, x_opt_pyomo, y_opt_pyomo = solve_flp(instance_name, False)

    # Encode those values into numpy arrays for easiness
    for idx in x_opt_pyomo:
        x_opt[idx] = x_opt_pyomo[idx].value
    for idx in y_opt_pyomo:
        y_opt[idx] = y_opt_pyomo[idx].value

    # Start rounding the values with a Greedy Rounding algorithm

    # Sort y in decreasing order
    # We build an index array
    y_opt_order = np.flip(np.argsort(y_opt))

    for j_idx in range(location_nb):
        # Get the j_idx index
        # Start with the facility location j with the best score (optimized by relaxation LP)
        j = y_opt_order[j_idx]

        # Value set to True, factory will be used
        y[j] = True

        # Sort x[:, j] in decreasing order.
        # Start with the supply assignation giving the best score (optimized by relaxation LP)
        x_opt_order = np.flip(np.argsort(x_opt[:, j]))
        for i_idx in range(customer_nb):
            i = x_opt_order[i_idx]
            # Checks if the facility capacity has not been reached yet
            # and checks if the customer demand is still not satisfied
            if np.sum(x[:, j]) < fac_capacity[j] and np.sum(x[i, :]) < customer_demand[i]:
                # Puts at maximum while not exceeding the capacity/demand
                x[i, j] = min(fac_capacity[j] - np.sum(x[:, j]), customer_demand[i] - np.sum(x[i, :]))

        # Checks if all customer demand has been satisfied
        continue_rounding = False
        for i in range(customer_nb):
            if np.sum(x[i, :]) < customer_demand[i]:
                continue_rounding = True

        # Returns the values if the customer demand has been satisfied
        if not continue_rounding:
            return x, y


def local_search_flp(x, y):
    pass
    # return (obj,x,y)


if __name__ == '__main__':
    test()
    # solve_flp('FLP-100-20-1.txt', False)  # test
    for f_name in os.listdir('./instances/'):
        x, y = initial_solution_flp(f_name)
        print(x, y)

