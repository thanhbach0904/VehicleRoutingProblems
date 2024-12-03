import math
import itertools
class CVRP_Greedy:
    def __init__(self, num_customers, num_vehicles, demands, cost_matrix, capacity):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.demands = demands
        self.cost_matrix = cost_matrix
        self.max_capacity = capacity
        self.routes = [[0,0] for _ in range(num_vehicles)]
        self.capacities = [0 for _ in range(num_vehicles)]
    def consider_insert_option(self,insert_position, vehicles_id):
        pass
    def solve(self):
        unchoosed_customer = [False]*self.num_customers
        for customer_id in range(self.num_customers):
            if not unchoosed_customer[customer_id]: #the customer is not visited by any vehicles 
                min_cost = float("inf")
                for vehicles_id in range(self.num_vehicles):
                    for insert_position in range(1,len(self.routes[vehicles_id]))
        pass


def read_input(file_content):
    lines = file_content.strip().split("\n")
    capacity = 0
    demands = []
    nodes = []
    parsing_nodes = False
    parsing_demands = False

    for line in lines:
        if line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            parsing_nodes = True
        elif line.startswith("DEMAND_SECTION"):
            parsing_nodes = False
            parsing_demands = True
        elif line.startswith("DEPOT_SECTION"):
            parsing_demands = False
        elif parsing_nodes:
            _, x, y = line.split()
            nodes.append((float(x), float(y)))
        elif parsing_demands:
            _, demand = line.split()
            demands.append(int(demand))
    
    return nodes, demands, capacity

def compute_cost_matrix(nodes):
    n = len(nodes)
    cost_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                cost_matrix[i][j] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return cost_matrix

if __name__ == "__main__":
    #input
    with open("E-n22-k4.vrp", "r") as file:
        nodes,demands, capacity = read_input(file.read())
        cost_matrix = compute_cost_matrix(nodes)
        print(cost_matrix)
        print(demands)
        print(capacity)
