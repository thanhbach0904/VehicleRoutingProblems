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