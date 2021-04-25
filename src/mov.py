import numpy as np


def factory_reassign(x, y, closed_f, open_f, capacity, demand, transport_cost):  # greedy reassign
    # close and open factories (reassign)
    for elem in closed_f:
        y[elem] = 1
    for elem in open_f:
        y[elem] = 0
    for c in open_f:
        x[:, c] = 0  # cancel transports for factories that are now closed

    demand_s_index = np.flip(np.argsort(demand))  # index of customers sorted by demand
    for i in demand_s_index:
        fcost_s_index = np.argsort(
            transport_cost[i, :])  # index of facilities sorted by the cost of transport to given customer
        for j in fcost_s_index:
            if y[j] == 1 and np.sum(x[:, j]) < capacity[j] and np.sum(x[i, :]) < demand[i]:
                x[i, j] = min(capacity[j] - np.sum(x[:, j]), demand[i] - np.sum(x[i, :]))
    return x, y


def assignment_mov_reassign(x, y, capacity, demand, random_customer, random_factories, transport_cost):
    # perform reassignment
    for elem in random_customer:
        left = demand[int(elem)] - np.sum(
            x[int(elem), :])  # what amount of unit is needed to fill the customer's demand
        while True:
            random_factoryA = np.random.choice(random_factories, replace=False)
            random_factoryB = np.random.choice(random_factories, replace=False)
            roomA = capacity[int(random_factoryA)] - np.sum(x[:, int(random_factoryA)])  # capacity left
            if random_factoryA != random_factoryB:
                roomB = capacity[int(random_factoryB)] - np.sum(x[:, int(random_factoryB)])  # capacity left
            else:
                roomB = 0
            if roomA + roomB >= left and roomA > 0:  # if the two (or one) factorie(s) selected can satisfy the demand
                break

        if random_factoryA != random_factoryB:
            if min(roomA, left) != max(left - roomB, 0):
                valueA = np.random.choice(np.arange(max(left - roomB, 0), min(roomA, left)))
            else:
                valueA = roomA
            valueB = left - valueA
            x[int(elem), int(random_factoryB)] = valueB
        else:
            valueA = left
        x[int(elem), int(random_factoryA)] = valueA

    return x, y


def assignment_mov_reassign_greedy(x, y, capacity, demand, random_customer, random_factories, transport_cost):
    demand_s_index = np.flip(np.argsort(demand))  # index of customers sorted by demand

    for i in demand_s_index:
        if i in random_customer:
            fcost_s_index = np.argsort(
                transport_cost[i, :])  # index of facilities sorted by the cost of transport to given customer
            for j in fcost_s_index:
                if j in random_factories and np.sum(x[:, j]) < capacity[j] and np.sum(x[i, :]) < demand[i]:
                    x[i, j] = min(capacity[j] - np.sum(x[:, j]), demand[i] - np.sum(x[i, :]))
    return x, y


def assignment_mov_reassign_greedy_random(x, y, capacity, demand, random_customer, random_factories, transport_cost):
    demand_s_index = np.flip(np.argsort(demand))  # index of customers sorted by demand

    for i in demand_s_index:
        if i in random_customer:
            fcost_s_index = np.argsort(
                transport_cost[i, :])  # index of facilities sorted by the cost of transport to given customer
            while np.sum(x[i, :]) < demand[i]:
                for j in fcost_s_index:
                    if j in random_factories and np.sum(x[:, j]) < capacity[j] and np.sum(x[i, :]) < demand[i]:
                        x[i, j] += int(
                            np.random.choice(1, min(capacity[j] - np.sum(x[:, j]), demand[i] - np.sum(x[i, :])))[0])
    return x, y


def factory_mov(x, y, capacity, demand, transport_cost):
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
        return factory_reassign(x, y, random_closed_factories, random_open_factories, capacity, demand, transport_cost)
    elif closed_factories.size == 0:
        return x, y
    else:
        return factory_mov(x, y, capacity, demand, transport_cost)


def assignment_mov_bis(x, y, capacity, demand, transport_cost):
    """
    second implementation of this method, for tweaking purpose
    :param x: initial sol
    :param y: initial sol
    :param capacity: param of the problem
    :param demand: param of the problem
    :return: optimized sol
    randomly select 2 customers,
    randomly select up to 2 factories per customers,
    Greedily reassign
    """
    random_customer = np.array([])

    # randomly select 2 customers
    for it in range(2):
        random_customer = np.append(random_customer, np.random.choice(x.shape[0]))

    random_factories = np.array([])
    # randomly select up to 2 factories per customer
    for elem in random_customer:
        factories = np.array(np.where(x[int(elem)] != 0)).flatten()
        for it in range(2 if factories.size > 1 else 1):
            if factories.size == 0:
                break
            random_pick = np.random.choice(factories, replace=False)
            factories = factories[factories != random_pick]
            random_factories = np.append(random_factories, random_pick)

    for elem in random_customer:
        for fac in random_factories:
            x[int(elem), int(fac)] = 0  # reinitialize assignments

    return assignment_mov_reassign_greedy(x, y, capacity, demand, random_customer, random_factories,
                                                 transport_cost)
