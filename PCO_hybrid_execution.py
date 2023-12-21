import time

def measure_performance(objective_function, *args, **kwargs):
    start_time = time.time()
    best_solution, best_value = hybrid_optimizer(objective_function, *args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return best_solution, best_value, execution_time

# Example Usage:
num_particles = 30
num_dimensions = 2
search_space = [-5, 5]
num_ants = 5
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5
num_iterations = 20
cognitive_param = 1.5
social_param = 1.5
inertia_weight = 0.7

best_solution, best_value, execution_time = measure_performance(objective_function, graph, num_particles, num_dimensions, search_space,
                                                                num_ants, alpha, beta, evaporation_rate, num_iterations, cognitive_param, social_param, inertia_weight, show_animation=False)

print("Best Solution:", best_solution)
print("Best Value:", best_value)
print("Execution Time:", execution_time, "seconds")