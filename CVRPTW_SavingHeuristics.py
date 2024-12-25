from dataclasses import dataclass
from typing import List, Tuple
import math
import os
import glob

class Customer:
    def __init__(self, id: int, x: float, y: float, demand: float, ready_time: float, due_date: float, service_time: float):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time


class Vehicle:
    def __init__(self, capacity: float):
        self.capacity = capacity
        self.route: List[Customer] = []
        self.total_demand: float = 0
        self.total_distance: float = 0
        self.current_time: float = 0  

class VRPTWSavings:
    def __init__(self, depot: Customer, customers: List[Customer], 
                 num_vehicles: int, vehicle_capacity: float):
        self.depot = depot
        self.customers = customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.saving_coefficient = 1
        self.vehicles: List[Vehicle] = []

    def euclidean_distance(self, c1: Customer, c2: Customer) -> float:
        return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

    def calculate_savings(self) -> List[Tuple[float, int, int]]:
        savings = []
        for i in range(len(self.customers)):
            for j in range(i+1, len(self.customers)):
                customer_i = self.customers[i]
                customer_j = self.customers[j]
                
                # Calculate savings by combining routes
                saving = (self.euclidean_distance(self.depot, customer_i) + 
                          self.euclidean_distance(self.depot, customer_j) - 
                          self.saving_coefficient * self.euclidean_distance(customer_i, customer_j))
                
                savings.append((saving, customer_i.id, customer_j.id))
        
        return sorted(savings, reverse=True)

    def is_feasible_insertion(self, vehicle: Vehicle, new_customers: List[Customer]) -> bool:
        total_new_demand = sum(customer.demand for customer in new_customers)
        new_total_demand = vehicle.total_demand + total_new_demand

        if new_total_demand > self.vehicle_capacity:
            return False

        # Check time window constraints
        current_time: float = 0
        prev_customer = self.depot
        for customer in vehicle.route + new_customers:
            travel_time = self.euclidean_distance(prev_customer, customer)
            arrival_time = current_time + travel_time
            
            if arrival_time > customer.due_date:
                return False

            current_time = max(arrival_time, customer.ready_time) + customer.service_time
            prev_customer = customer

        return True

    def solve(self) -> List[Vehicle]:
        #Initialize: each customer in a separate route
        self.vehicles = []
        for customer in self.customers:
            vehicle = Vehicle(self.vehicle_capacity)
            vehicle.route = [customer]
            vehicle.total_demand = customer.demand
            vehicle.total_distance = (self.euclidean_distance(self.depot, customer) * 2)
            vehicle.current_time = max(self.depot.ready_time + self.euclidean_distance(self.depot, customer), customer.ready_time) + customer.service_time
            self.vehicles.append(vehicle)

        #Get sorted savings
        savings = self.calculate_savings()

        #Merge routes based on savings
        for saving, customer_i_id, customer_j_id in savings:
            #Find vehicles containing these customers
            vehicle_i = next((v for v in self.vehicles if any(c.id == customer_i_id for c in v.route)), None)
            vehicle_j = next((v for v in self.vehicles if any(c.id == customer_j_id for c in v.route)), None)

            #Skip if same vehicle or vehicles don't exist
            if not vehicle_i or not vehicle_j or vehicle_i == vehicle_j:
                continue

            #Check if merging is feasible considering capacity and time windows
            if self.is_feasible_insertion(vehicle_i, vehicle_j.route):
                # Merge routes and update total demand and distance
                vehicle_i.route.extend(vehicle_j.route)
                vehicle_i.total_demand += vehicle_j.total_demand

                total_distance: float = 0
                current_time: float = 0
                prev_customer = self.depot

                for customer in vehicle_i.route:
                    travel_time = self.euclidean_distance(prev_customer, customer)
                    total_distance += travel_time

                    arrival_time = current_time + travel_time
                    current_time = max(arrival_time, customer.ready_time) + customer.service_time
                    prev_customer = customer

                total_distance += self.euclidean_distance(prev_customer, self.depot)
                vehicle_i.total_distance = total_distance
                vehicle_i.current_time = current_time

                self.vehicles.remove(vehicle_j)

        return self.vehicles

    def print_solution(self):
        valid_solution = True
        for i, vehicle in enumerate(self.vehicles, 1):
            print(f"\nVehicle {i}#:", end=" ")
            customer_count = 0
            for customer in vehicle.route:
                customer_count += 1
                print(f"{customer.id}", end=" ")

            if vehicle.total_demand > self.vehicle_capacity:
                print(f"  *** CAPACITY VIOLATION: Exceeds by {vehicle.total_demand - self.vehicle_capacity} ***")
                valid_solution = False

        if valid_solution:
            print("\nSolution is feasible.")
        else:
            print("WARNING: Solution contains capacity violations!")

        total_distance = sum(v.total_distance for v in self.vehicles)
        print(f"\nTotal cost: {total_distance:.2f}")



def read_input(file_path):

     with open(file_path, 'r') as f:
        lines = f.readlines()
    
        name = next(line.strip() for line in lines if line.strip())
        vehicle_section_start = next(i for i, line in enumerate(lines) 
                                if "VEHICLE" in line)
        vehicle_data = lines[vehicle_section_start + 2].split()
        max_vehicles = int(vehicle_data[0])
        vehicle_capacity = float(vehicle_data[1])
        
        
        customer_section_start = next(i for i, line in enumerate(lines) 
                                    if "CUSTOMER" in line)
        customer_start = customer_section_start + 3
        
        
        customers = []
        for line in lines[customer_start:]:
            
            if not line.strip():
                continue
                
            
            data = line.split()
            if len(data) < 7: 
                continue
                
            customer = Customer(
                id=int(data[0]),
                x=float(data[1]),
                y=float(data[2]),
                demand=float(data[3]),
                ready_time=float(data[4]),
                due_date=float(data[5]),
                service_time=float(data[6]))
            customers.append(customer)
        
        
        depot = customers[0]
        customers = customers[1:]  
        
        return (name,
            max_vehicles,
            vehicle_capacity,
            depot,
            customers)

def create_solutions_file(input_dir: str, output_file: str):
     with open(output_file, 'w') as f:
        f.write("Saving Heuristics Results\n")
        f.write("==========================\n\n")
        
        # Process all .txt files in the directory
        for input_file in glob.glob(os.path.join(input_dir, '*.txt')):
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            try:
                # Read input and create solver
                name, max_vehicles, vehicle_capacity, depot, customers = read_input(input_file)
                solver = VRPTWSavings(depot, customers, max_vehicles, vehicle_capacity)
                solution = solver.solve()
                
                f.write(f"{base_name}\n") 
                # Write vehicle routes
                for i, vehicle in enumerate(solution, 1):
                    f.write(f"Vehicle {i}#: ")
                    route_str = " ".join(str(customer.id) for customer in vehicle.route)
                    f.write(f"{route_str}\n")
            
                total_distance = sum(v.total_distance for v in solution)
                f.write(f"Total cost: {total_distance:.2f}\n\n")
                print(f"Processed {base_name}")
            except Exception as e:
                print(f"Error processing {base_name}: {str(e)}")
                f.write(f"Error processing {base_name}: {str(e)}\n\n")
        
        



if __name__ == "__main__":
    input_path = r"C:\Users\dmin\HUST\20241\Project1\test input\Vrp-Set-Solomon"
    output_file = os.path.join(r"C:\Users\dmin\HUST\20241\Project1", "comparison_results_VRPTW_SavingHeuristics.txt")
    create_solutions_file(input_path,output_file)