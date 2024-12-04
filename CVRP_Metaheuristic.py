import math
import random
import logging
from typing import List, Tuple, Dict

class AdvancedCVRPSolver:
    def __init__(self, num_customers: int, num_vehicles: int, 
                 demands: List[int], cost_matrix: List[List[float]], 
                 capacity: int):
        """
        Advanced Capacitated Vehicle Routing Problem Solver
        
        Args:
            num_customers: Total number of customers
            num_vehicles: Number of available vehicles
            demands: Demand for each customer
            cost_matrix: Distance/cost matrix between nodes
            capacity: Maximum capacity of each vehicle
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Problem parameters
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.demands = demands
        self.cost_matrix = cost_matrix
        self.max_capacity = capacity
        
        # Solution tracking
        self.routes = [[] for _ in range(num_vehicles)]
        self.vehicle_loads = [0] * num_vehicles
        self.total_cost = 0
        self.unserved_customers = set(range(1, num_customers))
        
        # Initialization logging
        self.logger.info(f"CVRP Problem Initialized")
        self.logger.info(f"Customers: {num_customers}")
        self.logger.info(f"Vehicles: {num_vehicles}")
        self.logger.info(f"Vehicle Capacity: {capacity}")
    
    def nearest_insertion_greedy(self) -> Dict:
        """
        Advanced Greedy Algorithm with Nearest Insertion Strategy
        
        Returns:
            Dictionary containing solution details
        """
        self.logger.info("Starting Nearest Insertion Greedy Algorithm")
        
        # Initialize routes with depot
        for i in range(self.num_vehicles):
            self.routes[i] = [0, 0]  # Start and end at depot
        
        while self.unserved_customers:
            best_insertion = self._find_best_insertion()
            
            if best_insertion is None:
                self.logger.warning("Cannot place remaining customers!")
                break
            
            customer, vehicle_idx, insert_pos, insertion_cost = best_insertion
            
            # Update route and tracking
            self.routes[vehicle_idx].insert(insert_pos, customer)
            self.vehicle_loads[vehicle_idx] += self.demands[customer]
            self.total_cost += insertion_cost
            self.unserved_customers.remove(customer)
            
            self.logger.info(f"Inserted Customer {customer} into Vehicle {vehicle_idx}")
        
        # Final logging
        self._log_solution_summary()
        
        return {
            "routes": self.routes,
            "total_cost": self.total_cost,
            "vehicle_loads": self.vehicle_loads
        }
    
    def simulated_annealing(self, initial_temp: float = 1000, 
                             cooling_rate: float = 0.95, 
                             iterations: int = 1000) -> Dict:
        """
        Simulated Annealing Metaheuristic for CVRP
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Rate of temperature reduction
            iterations: Number of iterations
        
        Returns:
            Dictionary containing best solution
        """
        self.logger.info("Starting Simulated Annealing Algorithm")
        
        # Initial solution using greedy
        current_solution = self.nearest_insertion_greedy()
        best_solution = current_solution.copy()
        
        temperature = initial_temp
        
        for i in range(iterations):
            # Generate neighborhood solution
            neighbor_solution = self._generate_neighbor()
            
            # Acceptance probability
            delta_cost = neighbor_solution['total_cost'] - current_solution['total_cost']
            
            # Metropolis criterion
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                current_solution = neighbor_solution
                
                # Update best solution if needed
                if current_solution['total_cost'] < best_solution['total_cost']:
                    best_solution = current_solution
            
            # Cool down
            temperature *= cooling_rate
            
            if i % 100 == 0:
                self.logger.info(f"Iteration {i}: Current Cost = {current_solution['total_cost']}")
        
        self.logger.info("Simulated Annealing Completed")
        return best_solution
    
    def _find_best_insertion(self):
        """Find best customer insertion across all vehicles"""
        best_insertion = None
        min_cost = float('inf')
        
        for customer in self.unserved_customers:
            for vehicle_idx in range(self.num_vehicles):
                for insert_pos in range(1, len(self.routes[vehicle_idx])):
                    # Check capacity constraint
                    if (self.vehicle_loads[vehicle_idx] + 
                        self.demands[customer] > self.max_capacity):
                        continue
                    
                    # Calculate insertion cost
                    prev_node = self.routes[vehicle_idx][insert_pos-1]
                    next_node = self.routes[vehicle_idx][insert_pos]
                    
                    insertion_cost = (
                        self.cost_matrix[prev_node][customer] + 
                        self.cost_matrix[customer][next_node] - 
                        self.cost_matrix[prev_node][next_node]
                    )
                    
                    if insertion_cost < min_cost:
                        best_insertion = (
                            customer, vehicle_idx, insert_pos, insertion_cost
                        )
                        min_cost = insertion_cost
        
        return best_insertion
    
    def _generate_neighbor(self):
        """
        Generate a neighboring solution by swapping customer assignments
        
        Returns:
            Neighboring solution dictionary
        """
        # Create a copy of current routes and loads
        neighbor_routes = [route.copy() for route in self.routes]
        neighbor_loads = self.vehicle_loads.copy()
        
        # Randomly select two different vehicles
        v1, v2 = random.sample(range(self.num_vehicles), 2)
        
        # Randomly select customers from these vehicles
        if len(neighbor_routes[v1]) > 2 and len(neighbor_routes[v2]) > 2:
            c1_idx = random.randint(1, len(neighbor_routes[v1])-2)
            c2_idx = random.randint(1, len(neighbor_routes[v2])-2)
            
            c1, c2 = neighbor_routes[v1][c1_idx], neighbor_routes[v2][c2_idx]
            
            # Verify capacity constraints
            if (neighbor_loads[v1] - self.demands[c1] + self.demands[c2] <= self.max_capacity and
                neighbor_loads[v2] - self.demands[c2] + self.demands[c1] <= self.max_capacity):
                
                # Swap customers
                neighbor_routes[v1][c1_idx], neighbor_routes[v2][c2_idx] = c2, c1
                
                # Update loads
                neighbor_loads[v1] = sum(self.demands[c] for c in neighbor_routes[v1][1:-1])
                neighbor_loads[v2] = sum(self.demands[c] for c in neighbor_routes[v2][1:-1])
        
        # Recalculate total cost
        total_cost = sum(
            sum(self.cost_matrix[neighbor_routes[i][j]][neighbor_routes[i][j+1]] 
                for j in range(len(neighbor_routes[i])-1))
            for i in range(self.num_vehicles)
        )
        
        return {
            'routes': neighbor_routes,
            'total_cost': total_cost,
            'vehicle_loads': neighbor_loads
        }
    
    def _log_solution_summary(self):
        """Log detailed information about the solution"""
        self.logger.info("Solution Summary:")
        self.logger.info(f"Total Route Cost: {self.total_cost}")
        self.logger.info("Vehicle Routes:")
        for i, route in enumerate(self.routes):
            self.logger.info(f"Vehicle {i}: {route}")
            self.logger.info(f"Vehicle Load: {self.vehicle_loads[i]}")
        
        unserved = len(self.unserved_customers)
        self.logger.info(f"Unserved Customers: {unserved}")

def read_vrp_file(filepath):
    """Read VRP file and extract problem parameters"""
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

def main():
    # Example usage
    nodes, demands, capacity = read_vrp_file("VehicleRoutingProblems\E-n22-k4.vrp")
    cost_matrix = compute_cost_matrix(nodes)
    
    solver = AdvancedCVRPSolver(
        num_customers=len(nodes),
        num_vehicles=4,
        demands=demands,
        cost_matrix=cost_matrix,
        capacity=capacity
    )
    
    # Choose solution method
    greedy_solution = solver.nearest_insertion_greedy()
    sa_solution = solver.simulated_annealing()
    print(greedy_solution)
    print(sa_solution)

if __name__ == "__main__":
    main()