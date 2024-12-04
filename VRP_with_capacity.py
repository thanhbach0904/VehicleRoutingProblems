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
        self.total_cost = 0
        self.unchoosed_customer = [False]*self.num_customers
    def consider_insert_option(self, insert_position, vehicles_id, customer_id):
        vehicle_route = self.routes[vehicles_id].copy()
        vehicle_route.insert(insert_position, customer_id)

        prev_node = vehicle_route[insert_position-1]
        next_node = vehicle_route[insert_position+1]
        cost = (self.cost_matrix[prev_node][customer_id] + 
                self.cost_matrix[customer_id][next_node] - 
                self.cost_matrix[prev_node][next_node])
        
        capacity_of_vehicle_if_insert = self.capacities[vehicles_id] + self.demands[customer_id]
        if capacity_of_vehicle_if_insert <= self.max_capacity:
            return cost
        else:
            return float("inf")
    def update(self,vehicle_id,customer_id,insert_position,cost):
        self.routes[vehicle_id].insert(insert_position,customer_id)
        self.capacities[vehicle_id] += self.demands[customer_id]
        self.total_cost += cost
    def check_if_any_customer_remain(self):
        for i in range(len(self.unchoosed_customer)):
            if self.unchoosed_customer[i] == False:
                return True
        return False
    def solve(self):
        while self.check_if_any_customer_remain():
            customers_served_this_iteration = False
            
            for customer_id in range(1, self.num_customers):
                if not self.unchoosed_customer[customer_id]:
                    min_cost = float("inf")
                    best_vehicle = None
                    best_position = None
                    for vehicles_id in range(self.num_vehicles):
                        for insert_position in range(1, len(self.routes[vehicles_id])):
                            cost = self.consider_insert_option(insert_position, vehicles_id, customer_id)
                            if cost < min_cost:
                                min_cost = cost
                                best_vehicle = vehicles_id
                                best_position = insert_position
                    if best_vehicle is not None:
                        self.update(best_vehicle, customer_id, best_position, min_cost)
                        self.unchoosed_customer[customer_id] = True
                        customers_served_this_iteration = True
                    else:
                        print(f"Customer {customer_id} cannot find a vehicle.")
            
        
            if not customers_served_this_iteration:
                print("Cannot serve remaining customers. Breaking the loop.")
                break
        
        
        unserved = [i for i in range(1, self.num_customers) if not self.unchoosed_customer[i]]
        if unserved:
            print(f"Unserved customers: {unserved}")
        
        return self.routes, self.total_cost, self.capacities


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
    with open("E\E-n101-k14.vrp", "r") as file:
        nodes,demands, capacity = read_input(file.read())
        n = len(nodes)
        k = 14
        cost_matrix = compute_cost_matrix(nodes)
        solver = CVRP_Greedy(n,k,demands,cost_matrix,capacity)
        print(solver.solve())
