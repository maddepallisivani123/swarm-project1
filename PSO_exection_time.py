import time

def measure_pso_performance(objective_function, *args, **kwargs):
    start_time = time.time()
    best_solution, best_value = pso_optimizer(objective_function, *args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return best_solution, best_value, execution_time

# Example Usage for PSO:
num_particles = 30
num_dimensions = 2
search_space = [-5, 5]
num_iterations = 100  # Increased iterations
cognitive_param = 1.5
social_param = 1.5
inertia_weight = 0.7

best_solution_pso, best_value_pso, execution_time_pso = measure_pso_performance(objective_function, num_particles, num_dimensions, search_space,
                                                                                   num_iterations, cognitive_param, social_param, inertia_weight, show_animation=False)

print("Best Solution (PSO):", best_solution_pso)
print("Best Value (PSO):", best_value_pso)
print("Execution Time (PSO):", execution_time_pso, "seconds")