from dataclasses import dataclass
from typing import List, Tuple
import math
import os
import glob
import time
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


class VRPTW_TOON:
    def __init__(self, depot: Customer, customers: List[Customer], 
                 num_vehicles: int, vehicle_capacity: float,params: Tuple[float, float, float]):
        self.depot = depot
        self.customers = customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.vehicles: List[Vehicle] = []
        #The parameters used for this heuristic, (c1, c2, c3), are: (0.4, 0.4, 0.2), (0, 1, 0), (0.5, 0.5, 0), and (0.3, 0.3, 0.4). 
        self.lambda_1, self.lambda_2, self.lambda_3 = params

    def euclidean_distance(self, c1: Customer, c2: Customer):
        return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    

    def calculate_cost(self, current: Customer, candidate: Customer, current_time: float) -> float:
        travel_time = self.euclidean_distance(current, candidate)
        arrival_time = current_time + travel_time
        T_ij = candidate.ready_time - arrival_time
        V_ij = candidate.due_date - arrival_time
        d_ij = travel_time

        return self.lambda_1 * d_ij + self.lambda_2 * max(0, T_ij) + self.lambda_3 * max(0, V_ij)

    def solve(self) -> List[Vehicle]:
        self.vehicles = []
        unserved_customers = set(self.customers)

        for _ in range(self.num_vehicles):
            vehicle = Vehicle(self.vehicle_capacity)
            current = self.depot
            current_time = 0

            while unserved_customers:
                best_customer = None
                best_cost = float('inf')
                for customer in unserved_customers:
                    if vehicle.total_demand + customer.demand > self.vehicle_capacity:
                        continue

                    cost = self.calculate_cost(current, customer, current_time)
                    if cost < best_cost:
                        travel_time = self.euclidean_distance(current, customer)
                        arrival_time = current_time + travel_time
                        if arrival_time <= customer.due_date:
                            best_customer = customer
                            best_cost = cost

                if best_customer:
                    vehicle.route.append(best_customer)
                    vehicle.total_demand += best_customer.demand
                    travel_time = self.euclidean_distance(current, best_customer)
                    current_time = max(current_time + travel_time, best_customer.ready_time) + best_customer.service_time
                    current = best_customer
                    unserved_customers.remove(best_customer)
                else:
                    break 

            if vehicle.route:
                vehicle.total_distance = sum(self.euclidean_distance(vehicle.route[i], vehicle.route[i + 1]) for i in range(len(vehicle.route) - 1)) + self.euclidean_distance(self.depot, vehicle.route[0]) + self.euclidean_distance(vehicle.route[-1], self.depot)
                self.vehicles.append(vehicle)

        return self.vehicles






def read_input(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

        
        try:
            name = next(line.strip() for line in lines if line.strip())
            vehicle_section_start = next(i for i, line in enumerate(lines) if "VEHICLE" in line)
            vehicle_data = lines[vehicle_section_start + 2].split()
            max_vehicles = int(vehicle_data[0])
            vehicle_capacity = float(vehicle_data[1])

            customer_section_start = next(i for i, line in enumerate(lines) if "CUSTOMER" in line)
            customer_start = customer_section_start + 3
        except (StopIteration, IndexError, ValueError) as e:
            raise ValueError(f"File format error in {file_path}: {str(e)}")

        customers = []
        for line in lines[customer_start:]:
            if not line.strip():
                continue
            data = line.split()
            if len(data) < 7:
                print(f"Skipping invalid customer data: {line.strip()}")
                continue

            customers.append(Customer(
                id=int(data[0]),
                x=float(data[1]),
                y=float(data[2]),
                demand=float(data[3]),
                ready_time=float(data[4]),
                due_date=float(data[5]),
                service_time=float(data[6])
            ))

        if not customers:
            raise ValueError(f"No valid customers found in file {file_path}")

        depot = customers[0]
        customers = customers[1:]  
        return name, max_vehicles, vehicle_capacity, depot, customers

def create_solutions_file(input_dir: str, output_dir: str):
    params_list = [
        (0.4, 0.4, 0.2),
        (0, 1, 0),
        (0.5, 0.5, 0),
        (0.3, 0.3, 0.4)]

    for params in params_list:
        param_name = f"params_{params[0]}_{params[1]}_{params[2]}"
        output_file = os.path.join(output_dir, f"Time_Oriented_Nearest_Neighbor_{param_name}_results.txt")

        with open(output_file, 'w') as f:
            f.write(f"Results for parameters {params}\n")
            f.write("=" * 30 + "\n\n")

            for input_file in glob.glob(os.path.join(input_dir, '*.txt')):
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                try:
                    name, max_vehicles, vehicle_capacity, depot, customers = read_input(input_file)
                    solver = VRPTW_TOON(depot, customers, max_vehicles, vehicle_capacity, params)
                    start_time = time.time()
                    solution = solver.solve()
                    end_time = time.time()

                    f.write(f"{base_name}\n")
                    for i, vehicle in enumerate(solution, 1):
                        f.write(f"Vehicle {i}#: ")
                        route_str = " ".join(str(customer.id) for customer in vehicle.route)
                        f.write(f"{route_str}\n")
                    
                    total_distance = sum(v.total_distance for v in solution)
                    f.write(f"Total cost: {total_distance:.2f}\n")
                    f.write(f"Time taken: {(end_time - start_time):.10f} seconds\n\n")
                    print(f"Processed {base_name} with parameters {params}")
                except Exception as e:
                    print(f"Error processing {base_name} with parameters {params}: {str(e)}")
                    f.write(f"Error processing {base_name}: {str(e)}\n\n")
        
        


def test_single_problem():
    input_file = r"C:\Users\dmin\HUST\20241\Project1\test input\Vrp-Set-Solomon\C101.txt"
    name, max_vehicles, vehicle_capacity, depot, customers = read_input(input_file)
    parameters = (0.5, 0.5, 0)
    solver = VRPTW_TOON(depot, customers, max_vehicles, vehicle_capacity,parameters)
    solution = solver.solve()
    for i,vehicle in enumerate(solution,1):
        print(f"Vehicle {i}#: ",end = "")
        route_str = " ".join(str(customer.id) for customer in vehicle.route)
        print(route_str)
    total_distance = sum(v.total_distance for v in solution)
    print(f"Total cost: {total_distance:.2f}")
if __name__ == "__main__":
    status = "w"
    if status == "w":
        input_path = r"C:\Users\dmin\HUST\20241\Project1\test input\Vrp-Set-Solomon"
        output_file = r"C:\Users\dmin\HUST\20241\Project1"
        create_solutions_file(input_path,output_file)
    else:
        test_single_problem()