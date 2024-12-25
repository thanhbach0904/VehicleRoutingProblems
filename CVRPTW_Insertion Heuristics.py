from typing import List, Tuple
import math
import os
import glob
import time
from collections import defaultdict
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


class VRPTW_InsertionHeuristics:
    def __init__(self, depot: Customer, customers: List[Customer], 
                 num_vehicles: int, vehicle_capacity: float,params: Tuple[float,float, float, float]):
        self.depot = depot
        self.customers = customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.vehicles: List[Vehicle] = []
        self.muy,self.alpha1,self.alpha2,self.lambda0 = params
    def euclidean_distance(self, c1: Customer, c2: Customer):
        return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

    def calculate_insertion_cost(self, i: Customer, u: Customer, j: Customer):
        b_ju = 0 #b_ju = max(b_u, u.ready_time) + u.service_time + self.euclidean_distance(u,j) || b_u = max()
        b_j = 0 #b_j = max(b_i , i.ready_time) + i.service_time + self.euclidean_distance(i,j)
        c11_iuj = self.euclidean_distance(i,u) + self.euclidean_distance(u,j) - self.muy * self.euclidean_distance(i,j)
        
        c12_iuj = b_ju - b_j
        c1_iuj = self.alpha1 * c11_iuj + self.alpha2 * c12_iuj
        c2_iuj = self.lambda0 * self.euclidean_distance(self.depot,u) - c1_iuj
        
        return c2_iuj

    def is_feasible_insertion(self, vehicle: Vehicle, u: Customer, i: Customer, j: Customer):
        
        # Calculate the new time at candidate and next if inserted
        arrival_time_at_candidate = max(i.ready_time + i.service_time + self.euclidean_distance(i, u), u.ready_time)
        if arrival_time_at_candidate > u.due_date:
            return False

        arrival_time_at_next = max(arrival_time_at_candidate + u.service_time + self.euclidean_distance(u, j), j.ready_time)
        if arrival_time_at_next > j.due_date:
            return False

        return True

    def solve(self):
        
        self.vehicles = []
        unserved_customers = set(self.customers)
        for _ in range(self.num_vehicles):
            vehicle = Vehicle(self.vehicle_capacity)
            current_time = 0
            vehicle.route = [self.depot,self.depot]
            begin_time_map = defaultdict(float)
            begin_time_map[0] = 0 #dictionary map from index -> begin time of route[index]
            while unserved_customers:
                best_insertion = None #this is the best insertion position
                best_cost = float("inf")
                for customer in unserved_customers:
                    if vehicle.total_demand + customer.demand > self.vehicle_capacity: #if this customer exceed the capacity constraint, then skip
                        continue
                    for insertion_pos in range(1,len(vehicle.route)):
                        i, j = vehicle.route[insertion_pos - 1] , vehicle.route[insertion_pos]
                        b_i = begin_time_map[insertion_pos - 1]
                        if self.is_feasible_insertion(vehicle, customer, i, j):
                            cost = self.calculate_insertion_cost(i,customer,j)
                            if cost < best_cost:
                                best_cost = cost
                                best_insertion = (insertion_pos,customer)
                if best_insertion:
                    insertion_pos, customer = best_insertion
                    vehicle.route.insert(insertion_pos, customer)
                    #begin_time_map[insertion_pos] = max(begin_time_map[insertion_pos -
                    unserved_customers.remove(customer)
                else:
                    break
            
            if vehicle.route:
                vehicle.total_distance = sum(self.euclidean_distance(vehicle.route[i], vehicle.route[i + 1]) for i in range(len(vehicle.route) - 1)) 
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

if __name__ == "__main__":
    input_file = r"C:\Users\dmin\HUST\20241\Project1\test input\Vrp-Set-Solomon\C101.txt"
    name, max_vehicles, vehicle_capacity, depot, customers = read_input(input_file)
    solver = VRPTW_InsertionHeuristics(depot, customers, max_vehicles, vehicle_capacity)
    solution = solver.solve()
    for i,vehicle in enumerate(solution,1):
        print(f"Vehicle {i}#: ",end = "")
        route_str = " ".join(str(customer.id) for customer in vehicle.route)
        print(route_str)
    total_distance = sum(v.total_distance for v in solution)
    print(f"Total cost: {total_distance:.2f}")