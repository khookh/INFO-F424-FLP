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

    factory_mov(x_greedy, y_greedy, fac_capacity, customer_demand, transport_cost)  # test
    assignment_mov(x_greedy, y_greedy, fac_capacity, customer_demand)  # test
    # Compute optimality gap between rounded solution and relaxed LP solution
    opening_cost = np.transpose(fac_opening_cost) @ y_greedy
    transport_cost = np.sum(np.multiply(transport_cost, x_greedy))

    rounded_cost = opening_cost + transport_cost

    cost_gap = opt_val - rounded_cost

    return cost_gap, x_greedy, y_greedy


def factory_reassign(x, y, closed_f, open_f, capacity, demand, cost):  # greedy reassign
    # close and open factories (reassign)
    for elem in closed_f:
        y[elem] = 1
    for elem in open_f:
        y[elem] = 0
    for c in open_f:
        x[:, c] = 0  # cancel transports for factories that are now closed

    demand_s_index = np.flip(np.argsort(demand))  # index of customers sorted by demand
    for i in demand_s_index:
        fcost_s_index = np.argsort(cost[i, :])  # index of facilities sorted by the cost of transport to given customer
        for j in fcost_s_index:
            if y[j] == 1 and np.sum(x[:, j]) < capacity[j] and np.sum(x[i, :]) < demand[i]:
                x[i, j] = min(capacity[j] - np.sum(x[:, j]), demand[i] - np.sum(x[i, :]))
    return x, y


def assignment_mov(x, y, capacity, demand):
    random_customer = np.array([])

    # randomly select up to 2 random customers
    for it in range(np.random.choice([1, 2])):
        random_customer = np.append(random_customer, np.random.choice(x.shape[0]))

    random_factories = np.array([])
    if random_factories.size == 0:
        return x, y
    # randomly select up to 2 factories per customer
    for elem in random_customer:
        factories = np.array(np.where(x[int(elem)] != 0)).flatten()
        previous_pick = -1
        random_pick = -1
        for it in range(2 if factories.size > 1 else 1):
            while random_pick == previous_pick:
                random_pick = np.random.choice(factories, replace=False)
            random_factories = np.append(random_factories, random_pick)
            x[int(elem), int(random_pick)] = 0  # reinitalize this assignment
            previous_pick = random_pick

    # perform reassignment
    for elem in random_customer:
        left = demand[int(elem)] - np.sum(x[int(elem), :]) # what amount of unit is needed to fill the customer's demand
        while True:
            random_factoryA = np.random.choice(random_factories, replace=False)
            random_factoryB = np.random.choice(random_factories, replace=False)
            roomA = capacity[int(random_factoryA)] - np.sum(x[:, int(random_factoryA)]) # capacity left
            if random_factoryA != random_factoryB:
                roomB = capacity[int(random_factoryB)] - np.sum(x[:, int(random_factoryB)]) # capacity left
            else:
                roomB = 0
            if roomA + roomB >= left and roomA > 0:  # if the two (or one) factorie(s) selected can satisfy the demand
                break

        if random_factoryA != random_factoryB:
            if left != roomA + roomB:
                valueA = np.random.choice(np.arange(max(left - roomB, 0), min(roomA, left)))
            else:
                valueA = roomA
            valueB = left - valueA
            x[int(elem), int(random_factoryB)] = valueB
        else:
            valueA = left
        x[int(elem), int(random_factoryA)] = valueA

    return x, y


def factory_mov(x, y, capacity, demand, cost):
    # closed_factory is the array containing the indexes of the closed factories
    closed_factories = np.array(np.where(y == False))

    # ""
    open_factories = np.array(np.where(y == True))

    # takes 0, 1 or 2 (depending on closed_factories.size) unique elements from the closed factories array
    random_closed_factories = np.random.choice(closed_factories[0], replace=False,
                                               size=closed_factories.size if closed_factories.size < 2 else 2)
    # ""
    random_open_factories = np.random.choice(open_factories[0], replace=False,
                                             size=closed_factories.size if closed_factories.size < 2 else 2)

    summed_capacity = 0
    for elem in random_closed_factories:
        summed_capacity += capacity[elem]

    summed_delivered = 0
    for elem in x:
        for fact in random_open_factories:
            summed_delivered += elem[fact]

    if summed_delivered < summed_capacity:
        # do reassign
        return factory_reassign(x, y, random_closed_factories, random_open_factories, capacity, demand, cost)
    elif closed_factories.size == 0:
        return x, y
    else:
        return factory_mov(x, y, capacity, demand, cost)


def local_search_flp(x, y):
    pass
    # return (obj,x,y)


if __name__ == '__main__':
    print(solve_flp(str(sys.argv[1]), False))  # test brute force

    for f_name in os.listdir('./Instances/'):
        cost_gap, x, y = initial_solution_flp(f_name)
        print(cost_gap)
