import multiprocessing
import time

import pyomo.environ as pyo
import numpy as np
import sys
import csv
import os

from mov import *
import threading
import queue

# Global instance parameters, will be set in the initial_solution function.

# Vector of capacity for each location
fac_capacity = None

# Vector of demand for each customer
customer_demand = None

# Matrix of transport cost for each supply unity between client and facility
transport_cost = None

# Vector of costs for each location openings
fac_opening_cost = None

# Number of customers
customer_nb = None

# Number of locations
location_nb = None

# Time criterion for the search algorithm, by default will stop after 10 mins of search.
time_criterion = 5 * 60

# Number of maximum iterations criterion
# !!! SET THIS VALUE TO NONE BEFORE SUBMITTING THE PROJECT !!!
max_iterations = None
# !!! SET THIS VALUE TO NONE BEFORE SUBMITTING THE PROJECT !!!

historic_fields = ['current_iter', 'current_cost', 'used_method', 'remaining_time']
historic_values = []


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


def verify_capacity(x, capacity):
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


def verify_demand(x, demand):
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
    print("brute cost :",
          compute_cost(np.asarray(list_x, dtype=np.float), np.asarray(list_y, dtype=np.float), fac_opening_cost,
                       transport_cost))  # temp
    return pyo.value(model.obj), list_x, list_y


def integrity_check(x, y):
    """
    :param x: sol
    :param y: sol
    :return: True if sol is acceptable
    Function used for test purpose : it tests the solution validity
    """
    closed_factories = np.array(np.where(y == False))
    for j in range(y.shape[0]):
        if y[j] != 0 and y[j] != 1:
            print("factory opening value not boolean")
            return False
        if np.sum(x[:, j]) > fac_capacity[j]:
            print("factory capacity overflow")
            return False
    for elem in closed_factories:
        if np.sum(x[:, elem]) != 0:
            print("closed factory has assignments")
            return False
    for i in range(x.shape[0]):
        if np.sum(x[i, :]) < customer_demand[i]:
            print("customer demand not fulfilled")
            return False
    return True


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


def compute_cost(x, y, fac_opening_cost, transport_cost):
    """
    Compute the cost function.
    :param x: The supply matrix (np.array)
    :param y: The construction vector (np.array)
    :param fac_opening_cost: The factory opening cost vector (np.array)
    :param transport_cost: The transport cost matrix (np.array)
    :return: the total cost (float)
    """
    opening_cost = np.sum(np.multiply(fac_opening_cost, y))
    transport_cost = np.sum(np.multiply(transport_cost, x))
    return opening_cost + transport_cost


def convert_instance_to_numpy(instance_param):
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

    return fac_capacity, customer_demand, transport_cost, fac_opening_cost, customer_nb, location_nb


def initial_solution_flp(instance_name):
    global fac_capacity, customer_demand, transport_cost, fac_opening_cost, customer_nb, location_nb

    instance_param = read_instance(instance_name)
    fac_capacity, customer_demand, transport_cost, fac_opening_cost, customer_nb, location_nb \
        = convert_instance_to_numpy(instance_param)

    print('There are {} customers.'.format(customer_nb))
    print('There are {} possible factories.'.format(location_nb))

    x_opt = np.zeros((customer_nb, location_nb), dtype=np.float)
    y_opt = np.zeros(location_nb, dtype=np.float)

    # Get relaxed LP solution values
    opt_val_pyomo, x_opt_pyomo, y_opt_pyomo = solve_flp(instance_name, True)

    # Encode those values into numpy arrays for easiness
    x_opt = np.asarray(x_opt_pyomo, dtype=np.float)
    y_opt = np.asarray(y_opt_pyomo, dtype=np.float)

    # Start rounding the values with a Greedy Rounding algorithm
    x_greedy, y_greedy = greedy_rounding(x_opt, y_opt, customer_nb, location_nb, customer_demand, fac_capacity)

    rounded_cost = compute_cost(x_greedy, y_greedy, fac_opening_cost, transport_cost)

    cost_gap = opt_val_pyomo - rounded_cost

    return cost_gap, x_greedy, y_greedy


# TODO: Check if we can change the function interface (as it might be tested automatically by the teacher)
def local_search_flp(x, y):
    """
    Performs a local search by iteratively passing to neighbor solutions while improving the cost at each iteration.
    :param x: Initial supply matrix obtained from relaxed LP + greedy rounding
    :param y: Initial factory building vector obtained from relaxed LP + greedy rounding
    :return: opt_sol, x_opt, y_opt
    """
    global fac_capacity, customer_demand, transport_cost, fac_opening_cost, customer_nb, location_nb, time_criterion, \
        max_iterations

    print('--------------------------\n'
          'Local search optimisation.\n'
          '--------------------------')

    def print_progress(current_cost, current_iter, max_iter, remaining_time, current_method, reg_history=False):
        if reg_history:
            historic_values.append([current_iter, current_cost, current_method.__name__, remaining_time / 100.0])
        print('', end='\r')
        print('Current cost: {} | Iterations: {}/{} | Time remaining: {:.1f}s ({:.2f}%) | Method: {}'
              .format(current_cost, iter_count, max_iterations if max_iterations is not None else 'None',
                      remaining_time_ms / 1000.0, 100.0 * (1 - (remaining_time / 1000.0) / (time_criterion)),
                      current_method.__name__), end='\r')

    iter_count = 0
    failed_iter_count = 0
    failed_iter_limit = 500
    failed_cycle_count = 0
    neighbor_evaluation_method, reseed_method = assignment_mov_bis, factory_mov

    remaining_time_ms = time_criterion * 1000
    initial_cost = compute_cost(x, y, fac_opening_cost, transport_cost)
    current_cost = initial_cost
    x_temp, y_temp = x.copy(), y.copy()
    # Register first entry
    print_progress(current_cost, iter_count, max_iterations, remaining_time_ms, neighbor_evaluation_method, True)

    print('Initial cost by greedy algorithm: {}'.format(initial_cost))
    while True:
        # Max iteration criteria
        if max_iterations is not None:
            if iter_count > max_iterations:
                break
        # Max execution time criteria
        if time_criterion is not None:
            if remaining_time_ms <= 0:
                break
        begin = time.time()

        if failed_iter_count >= failed_iter_limit:
            x_temp, y_temp = x.copy(), y.copy()
            for it in range(np.random.choice(np.arange(0, int(y.shape[0] / 10) + min(int(failed_cycle_count / 20), 5)))):
                x_temp, y_temp = np.random.choice([reseed_method, neighbor_evaluation_method])(x_temp, y_temp,
                                                                                               fac_capacity,
                                                                                               customer_demand,
                                                                                               transport_cost)
            failed_cycle_count += 1
            failed_iter_count = 0
        # Finds a random neighbor
        x_new, y_new = neighbor_evaluation_method(x_temp, y_temp, fac_capacity, customer_demand, transport_cost)

        # Computes the cost of the neighbor, if it optimises, then keep it as solution
        new_cost = compute_cost(x_new, y_new, fac_opening_cost, transport_cost)
        if new_cost < current_cost:
            print(new_cost, remaining_time_ms)
            print_progress(new_cost, iter_count, max_iterations, remaining_time_ms, neighbor_evaluation_method,
                           True)
            current_cost = new_cost
            failed_iter_count = 0
            failed_cycle_count = 0
            x, y = x_new, y_new
            x_temp, y_temp = x.copy(), y.copy()
        else:
            failed_iter_count += 1

        delta_time = (time.time() - begin) * 1000
        remaining_time_ms -= delta_time
        # if integrity_check(x, y) is False:  # for test purpose (temp)
        #    break
        iter_count += 1

    return current_cost, x, y


def save_history(path):
    with open(path, 'w') as f:
        write = csv.writer(f)
        write.writerow(historic_fields)
        write.writerows(historic_values)


def thread_work(task_queue):
    while not task_queue.empty():
        instance_name = task_queue.get()
        print('Start working on instance: {}'.format(instance_name))
        cost_greedy, x_greedy, y_greedy = initial_solution_flp(instance_name)

        cost_opt, x_opt, y_opt = local_search_flp(x_greedy, y_greedy)

        print('Task done: Greedy cost: {} | Optimal cost: {}'.format(cost_greedy, cost_opt))

        save_history(os.path.join('../Out', instance_name))


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python flp.py <instance_file_name>')
        sys.exit(0)

    instance_name = sys.argv[1]

    historic_values = []

    cost_greedy, x_greedy, y_greedy = initial_solution_flp(instance_name)
    solve_flp(instance_name, False)
    cost_opt, x_opt, y_opt = local_search_flp(x_greedy, y_greedy)

    print('Greedy cost: {} | Optimal cost: {}'.format(cost_greedy, cost_opt))

    save_history(os.path.join('../Out', instance_name))

    # task_queue = queue.Queue()
    # for instance_name in os.listdir('./Instances/'):
    #     task_queue.put(instance_name)
    #
    # thread_list = []
    #
    # for thread_id in range(multiprocessing.cpu_count()):
    #     thread_list.append(threading.Thread(target=thread_work, args=(task_queue,), daemon=True))
    #
    # for thread in thread_list:
    #     thread.join()
