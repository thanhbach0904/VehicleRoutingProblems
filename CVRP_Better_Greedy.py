import math
import random
import logging
from typing import List, Tuple, Dict
import time
import glob
import os

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
        
        #self.logger.info("Starting insert process.")
        
        
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
            
            #self.logger.info(f"Inserted Customer {customer} into Vehicle {vehicle_idx}")
        
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
            print(f'Problem code : n = {self.num_customers}, k = {self.num_vehicles}')
def read_input(file_content):
    lines = file_content.strip().split("\n")
    capacity = 0
    demands = []
    nodes = []
    name = ""
    edge_weight_type = ""
    parsing_nodes = False
    parsing_demands = False

    for line in lines:
        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
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

    return name, edge_weight_type, nodes, demands, capacity

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

def create_solutions_file(input_dir, output_file):
    with open(output_file, "w") as output:
        for input_file in glob.glob(os.path.join(input_dir, "*.vrp")):
            with open(input_file, "r") as file:
                name, edge_weight_type, nodes, demands, capacity = read_input(file.read())

                if edge_weight_type != "EUC_2D":
                    print(f"Skipping {name}: Unsupported EDGE_WEIGHT_TYPE {edge_weight_type}")
                    continue

                num_customers = len(nodes)
                num_vehicles = int(name.split("-k")[1])  
                cost_matrix = compute_cost_matrix(nodes)

                solver = CVRP_Better_Greedy(num_customers, num_vehicles, demands, cost_matrix, capacity)
                start_time = time.time()
                solutions = solver.nearest_insertion_greedy()
                end_time = time.time()
                routes,total_cost,total_capacities = solutions["routes"],solutions["total_cost"],solutions["vehicle_loads"]

                output.write(f"{name}\n")
                output.write("=" * 30 + "\n")
                for i, route in enumerate(routes):
                    output.write(f"Vehicle {i + 1}: {route}\n")
                output.write(f"Total cost of the solution: {total_cost}\n")
                output.write(f"Total capacities of the solution: {total_capacities}\n")
                output.write(f"Time taken by the algorithm: {end_time - start_time:.10f} seconds\n")
                output.write("\n")

if __name__ == "__main__":
    input_dir = r"C:\Users\dmin\HUST\20241\Project1\test input\A"
    output_file = r"C:\Users\dmin\HUST\20241\Project1\better_greedy_result_set_A.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    create_solutions_file(input_dir, output_file)