"""
Simple Genetic Algorithm Parameter Example
This demonstrates how different parameters affect GA performance
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class SimpleGA:
    """A simple genetic algorithm to demonstrate parameter effects"""
    
    def __init__(self, pop_size: int, mut_rate: float, mut_sigma: float, elite_size: int):
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.mut_sigma = mut_sigma
        self.elite_size = elite_size
        
    def create_individual(self) -> np.ndarray:
        """Create a random individual (solution)"""
        return np.random.uniform(-5, 5, 10)  # 10-dimensional solution
    
    def fitness(self, individual: np.ndarray) -> float:
        """Calculate fitness (higher is better) - Rastrigin function"""
        n = len(individual)
        A = 10
        return A * n + sum(individual**2 - A * np.cos(2 * np.pi * individual))
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutate an individual"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mut_rate:
                # Gaussian mutation with sigma
                mutated[i] += np.random.normal(0, self.mut_sigma)
        return mutated
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Crossover two parents to create a child"""
        child = np.zeros_like(parent1)
        for i in range(len(child)):
            if np.random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child
    
    def select_parent(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        """Select a parent using tournament selection"""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]  # Lower is better for Rastrigin
        return population[winner_idx]
    
    def evolve(self, num_generations: int) -> Tuple[List[float], List[float]]:
        """Run the genetic algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_fitnesses = []
        avg_fitnesses = []
        
        for generation in range(num_generations):
            # Calculate fitnesses
            fitnesses = [self.fitness(ind) for ind in population]
            
            # Sort by fitness (lower is better for Rastrigin)
            sorted_indices = np.argsort(fitnesses)
            population = [population[i] for i in sorted_indices]
            fitnesses = [fitnesses[i] for i in sorted_indices]
            
            # Record statistics
            best_fitnesses.append(fitnesses[0])
            avg_fitnesses.append(np.mean(fitnesses))
            
            # Elitism: keep best individuals
            new_population = population[:self.elite_size]
            
            # Generate new individuals
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.select_parent(population, fitnesses)
                parent2 = self.select_parent(population, fitnesses)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_fitnesses, avg_fitnesses

def demonstrate_parameters():
    """Demonstrate how different parameters affect GA performance"""
    
    # Test different parameter configurations
    configs = [
        {"name": "Low Mutation", "pop_size": 50, "mut_rate": 0.1, "mut_sigma": 0.1, "elite_size": 5},
        {"name": "High Mutation", "pop_size": 50, "mut_rate": 0.8, "mut_sigma": 2.0, "elite_size": 5},
        {"name": "Small Population", "pop_size": 20, "mut_rate": 0.3, "mut_sigma": 0.5, "elite_size": 2},
        {"name": "Large Population", "pop_size": 100, "mut_rate": 0.3, "mut_sigma": 0.5, "elite_size": 10},
        {"name": "High Elite", "pop_size": 50, "mut_rate": 0.3, "mut_sigma": 0.5, "elite_size": 20},
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, config in enumerate(configs):
        ga = SimpleGA(**{k: v for k, v in config.items() if k != 'name'})
        best_fitnesses, avg_fitnesses = ga.evolve(50)
        
        plt.subplot(2, 3, i+1)
        plt.plot(best_fitnesses, label='Best', linewidth=2)
        plt.plot(avg_fitnesses, label='Average', alpha=0.7)
        plt.title(f"{config['name']}\nPop: {config['pop_size']}, Mut: {config['mut_rate']:.1f}, Sigma: {config['mut_sigma']:.1f}")
        plt.xlabel('Generation')
        plt.ylabel('Fitness (lower is better)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ga_parameter_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Parameter Demonstration Complete!")
    print("Generated: ga_parameter_demo.png")
    print("\nKey Insights:")
    print("1. Low mutation: Converges quickly but may get stuck in local optima")
    print("2. High mutation: More exploration, slower convergence")
    print("3. Small population: Faster but less diverse")
    print("4. Large population: More diverse but slower")
    print("5. High elite: Preserves good solutions but reduces diversity")

def explain_parameters():
    """Print educational explanation of GA parameters"""
    
    print("=== GENETIC ALGORITHM PARAMETERS EXPLAINED ===\n")
    
    print("1. POPULATION SIZE")
    print("   - What it does: Number of candidate solutions maintained")
    print("   - Trade-off: Larger = more diversity but slower convergence")
    print("   - Analogy: Like having more students in a class - more ideas but harder to manage")
    print("   - Your finding: Smaller populations worked better (unusual!)\n")
    
    print("2. MUTATION RATE")
    print("   - What it does: Probability that a gene gets randomly changed")
    print("   - Trade-off: Higher = more exploration but can disrupt good solutions")
    print("   - Analogy: Like proofreading - too little and you miss errors, too much and you break good text")
    print("   - Your finding: Moderate rates (0.3-0.6) work best\n")
    
    print("3. MUTATION SIGMA")
    print("   - What it does: Magnitude of random changes during mutation")
    print("   - Trade-off: Higher = larger jumps but can overshoot the target")
    print("   - Analogy: Like step size when walking - big steps cover ground fast but might miss the target")
    print("   - Your finding: Higher values are better (strongest correlation!)\n")
    
    print("4. ELITE SIZE")
    print("   - What it does: Number of best individuals preserved unchanged")
    print("   - Trade-off: Higher = preserves good solutions but reduces diversity")
    print("   - Analogy: Like keeping the best students from graduating - they stay but take up spots")
    print("   - Your finding: 10 seems to work well\n")
    
    print("5. RANK DEPTH (Tournament Size)")
    print("   - What it does: How many individuals compete in selection tournaments")
    print("   - Trade-off: Higher = more selective but less diversity")
    print("   - Analogy: Like job interviews - more candidates = more selective hiring")
    print("   - Your finding: Lower values work better (less selective pressure)\n")
    
    print("6. NUMBER OF GENERATIONS")
    print("   - What it does: How long the algorithm runs")
    print("   - Trade-off: More = better solutions but more computation time")
    print("   - Analogy: Like studying - more time usually means better results")
    print("   - Your finding: More generations help (as expected)\n")

if __name__ == "__main__":
    print("Running Genetic Algorithm Parameter Demonstration...")
    demonstrate_parameters()
    print("\n" + "="*50 + "\n")
    explain_parameters() 