import math
import random
import logging
from typing import List, Tuple, Dict
import time
class CVRP_Better_Greedy:
    def __init__(self, num_customers: int, num_vehicles: int, demands: List[int], cost_matrix: List[List[float]], capacity: int):

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.demands = demands
        self.cost_matrix = cost_matrix
        self.max_capacity = capacity
        
        
        self.routes : List = [[] for _ in range(num_vehicles)]
        self.vehicle_loads = [0] * num_vehicles
        self.total_cost = 0
        self.unserved_customers = set(range(1, num_customers))
        
        
        self.logger.info(f"CVRP Problem Initialized")
        self.logger.info(f"Customers: {num_customers}")
        self.logger.info(f"Vehicles: {num_vehicles}")
        self.logger.info(f"Vehicle Capacity: {capacity}")
    
    def nearest_insertion_greedy(self) -> Dict:
        
        self.logger.info("Starting insert process.")
        
        
        for i in range(self.num_vehicles):
            self.routes[i] = [0, 0]  
        
        while self.unserved_customers:
            best_insertion = self.find_best_insertion()
            
            if best_insertion is None:
                self.logger.warning("No insertion can be found.")
                break
            
            customer, vehicle_idx, insert_pos, insertion_cost = best_insertion
            
            
            self.routes[vehicle_idx].insert(insert_pos, customer)
            self.vehicle_loads[vehicle_idx] += self.demands[customer]
            self.total_cost += insertion_cost
            self.unserved_customers.remove(customer)
            
            self.logger.info(f"Inserted Customer {customer} into Vehicle {vehicle_idx}")
        
        self._log_solution_summary()
        
        return {"routes": self.routes, "total_cost": self.total_cost, "vehicle_loads": self.vehicle_loads}
    
    
    
    def find_best_insertion(self):
        
        best_insertion = None
        min_cost = float('inf')
        
        for customer in self.unserved_customers:
            for vehicle_idx in range(self.num_vehicles):
                for insert_pos in range(1, len(self.routes[vehicle_idx])):
                    
                    if (self.vehicle_loads[vehicle_idx] + 
                        self.demands[customer] > self.max_capacity):
                        continue #not consider this option because vehicles_loads is exceeded
                    
                    prev_node = self.routes[vehicle_idx][insert_pos-1]
                    next_node = self.routes[vehicle_idx][insert_pos]
                    
                    insertion_cost = (
                        self.cost_matrix[prev_node][customer] + 
                        self.cost_matrix[customer][next_node] - 
                        self.cost_matrix[prev_node][next_node])
                    
                    if insertion_cost < min_cost:
                        best_insertion = (customer, vehicle_idx, insert_pos, insertion_cost)
                        min_cost = insertion_cost
        
        return best_insertion
    

    
    def _log_solution_summary(self):
        
        self.logger.info("Solution Summary:")
        self.logger.info("Vehicle Routes:")
        for i, route in enumerate(self.routes):
            self.logger.info(f"Vehicle {i}: {route}")
        self.logger.info(f"Capacites of vehicles: {self.vehicle_loads}")
        self.logger.info(f"Total Route Cost: {self.total_cost}")
        unserved = len(self.unserved_customers)
        if unserved == 0:
            self.logger.info("No customers left.")
        else:
            self.logger.info(f"Number of unserved customers: {unserved}.")

def read_vrp_file(filepath):
    
    with open(filepath, 'r') as file:
        file_content = file.read()
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
    nodes, demands, capacity = read_vrp_file("test input\E\E-n101-k14.vrp")
    cost_matrix = compute_cost_matrix(nodes)
    
    solver = CVRP_Better_Greedy(num_customers=len(nodes), num_vehicles=14, demands=demands, cost_matrix=cost_matrix, capacity=capacity)
    start_time = time.time()
    greedy_solution = solver.nearest_insertion_greedy()
    end_time = time.time()
    print(f"Greedy solution found in {end_time - start_time} seconds")