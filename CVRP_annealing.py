import math
import random
import logging
from typing import List, Tuple, Dict
import time
class AdvancedCVRPSolver:
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
        
        return {"routes": self.routes,"total_cost": self.total_cost,"vehicle_loads": self.vehicle_loads}
    
    def simulated_annealing(self, initial_temp: float = 1000, cooling_rate: float = 0.9, iterations: int = 100000) -> Dict:
        
        
        # Initial solution using greedy
        current_solution = self.nearest_insertion_greedy()
        best_solution = current_solution.copy()
        
        temperature = initial_temp
        
        for i in range(iterations):
            #generate neighbor solution by randomly swap 2 customers between 2 vehicles
            neighbor_solution = self.generate_neighbor(option= "swap_2_cities")
            
            #criteria for changing solution
            delta_cost = neighbor_solution['total_cost'] - current_solution['total_cost']
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature): #only move if the new solution is better or take risks with probability
                current_solution = neighbor_solution
                if current_solution['total_cost'] < best_solution['total_cost']:
                    best_solution = current_solution
            
            
            temperature *= cooling_rate
            
            if i % 10000 == 0:
                self.logger.info(f"Iteration {i}: Current Cost = {current_solution['total_cost']}")
        
        self.logger.info(f"Simulated Annealing completed after {iterations} iterations.")
        return best_solution
    
    def find_best_insertion(self):
        best_insertion = None
        min_cost = float('inf')
        
        for customer in self.unserved_customers:
            for vehicle_idx in range(self.num_vehicles):
                for insert_pos in range(1, len(self.routes[vehicle_idx])):
                    
                    if (self.vehicle_loads[vehicle_idx] + 
                        self.demands[customer] > self.max_capacity):
                        continue
                    
                    
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
    
    def generate_neighbor(self,option = "swap_2_cities"):
        if option == "swap_2_cities": #hoan vi 2 thanh pho ngau nhien tu 2 xe ngau nhien
        
            neighbor_routes = [route.copy() for route in self.routes]
            neighbor_loads = self.vehicle_loads.copy()
            
            
            v1, v2 = random.sample(range(self.num_vehicles), 2)
            
            
            if len(neighbor_routes[v1]) > 2 and len(neighbor_routes[v2]) > 2:
                c1_idx = random.randint(1, len(neighbor_routes[v1])-2)
                c2_idx = random.randint(1, len(neighbor_routes[v2])-2)
                
                c1, c2 = neighbor_routes[v1][c1_idx], neighbor_routes[v2][c2_idx]
                
                
                if (neighbor_loads[v1] - self.demands[c1] + self.demands[c2] <= self.max_capacity and neighbor_loads[v2] - self.demands[c2] + self.demands[c1] <= self.max_capacity):
                    
                    neighbor_routes[v1][c1_idx], neighbor_routes[v2][c2_idx] = c2, c1 #swap if the swap does not cause capacity limit in both vehicles
                    
                    neighbor_loads[v1] = sum(self.demands[c] for c in neighbor_routes[v1][1:-1])
                    neighbor_loads[v2] = sum(self.demands[c] for c in neighbor_routes[v2][1:-1])
            
            
            total_cost = sum(
                sum(self.cost_matrix[neighbor_routes[i][j]][neighbor_routes[i][j+1]] 
                    for j in range(len(neighbor_routes[i])-1))
                for i in range(self.num_vehicles)
            )
            
            return {'routes': neighbor_routes, 'total_cost': total_cost, 'vehicle_loads': neighbor_loads}
        elif option == "reverse_sub_route_between_2_cities": #chon 2 thanh pho ngau nhien c1,c2 trong cung 1 route roi dao nguoc sub-route tu c1->c2
            #chon vehicle ngau nhien, chon thanh pho c1,c2 ngau nhien thuoc vehicle nay sao cho c1_idx < c2_idx. Dao nguoc sub-route nhu sau : routes[vehicle_id][c1_idx + 1 : c2_idx] = routes[vehicle_id][c1_idx + 1 : c2_idx][::-1]
            pass
        elif option == "insert_sub_route_to_another_pos": #chon 1 hanh trinh con roi chen vao vi tri ngau nhien
            #chon hanh trinh con tu vehicle co nhieu customer nhat -> vehicle co it customer nhat
            pass
        elif option == "insert_random_city_to_new_pos": #chon 1 thanh pho tu vehicle phuc vu nhieu khach -> vehicle phuc vu it khach
            neighbor_routes = [route.copy() for route in self.routes]
            neighbor_loads = self.vehicle_loads.copy()
            highest_customer_served_vehicle, lowest_customer_served_vehicle = None,None
            min_cus,max_cus = float("inf"), - float("inf")
            for idx,route in enumerate(neighbor_routes):
                if len(route) > max_cus:
                    max_cus = len(route)
                    highest_customer_served_vehicle = idx
                if len(route) < min_cus:
                    min_cus = len(route)
                    lowest_customer_served_vehicle = idx
            #potential approach : choose the vehicle which has the highest cost, remove the city that reduce the cost as much as possible and insert it into the vehicles with lowest cost.
            #chose a city from highest customer served vehicle
            c_idx = random.randint(1, len(neighbor_routes[highest_customer_served_vehicle]) - 2)
            city = neighbor_routes[highest_customer_served_vehicle][c_idx]
            if neighbor_loads[highest_customer_served_vehicle] >= self.demands[city] and neighbor_loads[lowest_customer_served_vehicle] + self.demands[city] <= self.max_capacity:
                new_neighbor_route = neighbor_routes[highest_customer_served_vehicle][:c_idx] + neighbor_routes[highest_customer_served_vehicle][c_idx + 1:] #remove the chosen city from highest 
                neighbor_routes[highest_customer_served_vehicle] = new_neighbor_route
                neighbor_loads[highest_customer_served_vehicle] -= self.demands[city] #update the loads of the highest city
                
                insert_pos = random.randint(1,len(neighbor_routes[lowest_customer_served_vehicle]) - 1)
                neighbor_routes[lowest_customer_served_vehicle].insert(insert_pos, city) #insert the chosen city to the lower city
                neighbor_loads[lowest_customer_served_vehicle] += self.demands[city] 

            total_cost = sum(
                sum(self.cost_matrix[neighbor_routes[i][j]][neighbor_routes[i][j+1]] 
                    for j in range(len(neighbor_routes[i])-1))
                for i in range(self.num_vehicles)
            )
            
            return {'routes': neighbor_routes, 'total_cost': total_cost, 'vehicle_loads': neighbor_loads}

        else:
            raise ValueError("Invalid option for generating neighbor. Can only choose options : [swap_2_cities, reverse_sub_route_between_2_cities, insert_sub_route_to_another_pos, insert_random_city_to_new_pos]")
    
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
    solver = AdvancedCVRPSolver(num_customers=len(nodes), num_vehicles=14, demands=demands, cost_matrix=cost_matrix, capacity=capacity)
    
    start_time = time.time()
    annealing_solution = solver.simulated_annealing()
    end_time = time.time()
    for i,route in enumerate(annealing_solution['routes']):
        print(f"Vehicles {i+1}: {route}")
    print(f"Total cost : {annealing_solution['total_cost']}")
    print(f"Capacites of vehicles : {annealing_solution['vehicle_loads']}")
    print(f"Simulated Annealing solution found in {end_time - start_time} seconds")