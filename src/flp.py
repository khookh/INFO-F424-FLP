import pyomo.environ as pyo
import numpy as np
import sys
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


def verif_capacity(x, capacity):
    delivered = [0] * len(capacity)
    for elem in x:
        count = 0
        for elem2 in elem:
            delivered[count] += elem2
            count += 1
    for i in range(len(capacity)):
        # print(delivered[i],capacity[i])
        if delivered[i] > capacity[i]:
            print("solution not valid : capacity overflow")


def verif_demand(x, demand):
    count = 0
    for elem in x:
        somme = 0
        for elem2 in elem:
            somme += elem2
        # print(somme, demand[count])
        if somme < demand[count]:
            print("solution not valid : demand isn't fulfilled")
        count += 1


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

    list_x = [[0 for x in range(len(instance_param[2]))] for y in range(len(instance_param[1]))]
    list_y = [0 for x in range(len(instance_param[2]))]

    for i in model.I:
        for j in model.J:
            list_x[i][j] = model.x[i, j].value
    for j in model.J:
        list_y[j] = model.y[j].value

    return pyo.value(model.obj), list_x, list_y


def greedy_rounding(x_opt, y_opt, customer_nb, location_nb, customer_demand, fac_capacity):
    # Initialize integer solution values
    x = np.zeros((customer_nb, location_nb), dtype=np.int)
    y = np.zeros(location_nb, dtype=np.bool)

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


def initial_solution_flp(instance_name):
    instance_param = read_instance(instance_name)
    customer_nb = len(instance_param[1])
    location_nb = len(instance_param[2])
    customer_demand = np.array(list(instance_param[1].values()))
    fac_opening_cost = np.array(list(instance_param[0].values()))
    fac_capacity = np.array(list(instance_param[2].values()))

    # Convert dict to 2D numpy array
    transport_cost = instance_param[3]
    transport_cost_np = np.zeros((customer_nb, location_nb), dtype=np.int)
    i, j = zip(*transport_cost.keys())
    np.add.at(transport_cost_np, tuple((i, j)), tuple(transport_cost.values()))
    transport_cost = transport_cost_np

    print('There are {} customers.'.format(customer_nb))
    print('There are {} possible factories.'.format(location_nb))

    x_opt = np.zeros((customer_nb, location_nb), dtype=np.float)
    y_opt = np.zeros(location_nb, dtype=np.float)

    # Get relaxed LP solution values
    opt_val, x_opt_pyomo, y_opt_pyomo = solve_flp(instance_name, True)

    # Encode those values into numpy arrays for easiness
    x_opt = np.asarray(x_opt_pyomo, dtype=np.float)
    y_opt = np.asarray(y_opt_pyomo, dtype=np.float)

    # Start rounding the values with a Greedy Rounding algorithm
    x_greedy, y_greedy = greedy_rounding(x_opt, y_opt, customer_nb, location_nb, customer_demand, fac_capacity)

    # Compute optimality gap between rounded solution and relaxed LP solution
    opening_cost = np.transpose(fac_opening_cost) @ y_greedy
    transport_cost = np.sum(np.multiply(transport_cost, x_greedy))

    rounded_cost = opening_cost + transport_cost

    cost_gap = opt_val - rounded_cost

    return cost_gap, x_greedy, y_greedy


def local_search_flp(x, y):
    pass
    # return (obj,x,y)


if __name__ == '__main__':
    print(solve_flp(str(sys.argv[1]), False))  # test brute force

    for f_name in os.listdir('./Instances/'):
        cost_gap, x, y = initial_solution_flp(f_name)
        print(cost_gap)
