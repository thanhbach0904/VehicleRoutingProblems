import math
import time
import os
import glob

class CVRP_Greedy:
    def __init__(self, num_customers, num_vehicles, demands, cost_matrix, capacity):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.demands = demands
        self.cost_matrix = cost_matrix
        self.max_capacity = capacity
        self.routes = [[0, 0] for _ in range(num_vehicles)]
        self.capacities = [0 for _ in range(num_vehicles)]
        self.total_cost = 0
        self.unchoosed_customer = [False] * self.num_customers

    def consider_insert_option(self, insert_position, vehicles_id, customer_id):
        vehicle_route = self.routes[vehicles_id].copy()
        vehicle_route.insert(insert_position, customer_id)

        prev_node = vehicle_route[insert_position - 1]
        next_node = vehicle_route[insert_position + 1]
        cost = (
            self.cost_matrix[prev_node][customer_id]
            + self.cost_matrix[customer_id][next_node]
            - self.cost_matrix[prev_node][next_node]
        )

        capacity_of_vehicle_if_insert = self.capacities[vehicles_id] + self.demands[customer_id]
        if capacity_of_vehicle_if_insert <= self.max_capacity:
            return cost
        else:
            return float("inf")

    def update(self, vehicle_id, customer_id, insert_position, cost):
        self.routes[vehicle_id].insert(insert_position, customer_id)
        self.capacities[vehicle_id] += self.demands[customer_id]
        self.total_cost += cost

    def check_if_any_customer_remain(self):
        for i in range(len(self.unchoosed_customer)):
            if not self.unchoosed_customer[i]:
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
            print(f'Problem code : n = {self.num_customers}, k = {self.num_vehicles}')

        return self.routes, self.total_cost, self.capacities

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

                solver = CVRP_Greedy(num_customers, num_vehicles, demands, cost_matrix, capacity)
                start_time = time.time()
                routes, total_cost, total_capacities = solver.solve()
                end_time = time.time()

                output.write(f"{name}\n")
                output.write("=" * 30 + "\n")
                for i, route in enumerate(routes):
                    output.write(f"Vehicle {i + 1}: {route}\n")
                output.write(f"Total cost of the solution: {total_cost}\n")
                output.write(f"Total capacities of the solution: {total_capacities}\n")
                output.write(f"Time taken by the algorithm: {end_time - start_time:.10f} seconds\n")
                output.write("\n")

if __name__ == "__main__":
    input_dir = r"C:\Users\dmin\HUST\20241\Project1\test input\E"
    output_file = r"C:\Users\dmin\HUST\20241\Project1\naive_greedy_result_set_E.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    create_solutions_file(input_dir, output_file)
