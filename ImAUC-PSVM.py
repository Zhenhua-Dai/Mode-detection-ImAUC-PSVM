import numpy as np
from sklearn.cluster import KMeans
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def read_data(file_path):
    df = pd.read_csv(file_path)
    # Convert string representations of lists into actual lists
    df['Tokens'] = df['Tokens'].apply(eval)
    df['Tags'] = df['Tags'].apply(eval)
    df['Polarities'] = df['Polarities'].apply(eval)
    return df

file_path = 'data\\train.csv'
data = read_data(file_path)

  
class PSVM:
    def __init__(self, gamma, epochs, batch_size, c_plus=None, c_minus=None):
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.c_plus = c_plus
        self.c_minus = c_minus

class PSVM:
    def __init__(self, gamma, epochs, batch_size, c_plus=None, c_minus=None):
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.c_plus = c_plus
        self.c_minus = c_minus

    def fit(self, X, y):
        # Feature mapping and bias augmentation
        phi_X = np.hstack((X, np.ones((X.shape[0], 1))))  # φ(x) | 1

        if self.c_plus is None or self.c_minus is None:
            # Calculate means for positive and negative classes
            self.c_plus = np.mean(phi_X[y == 1], axis=0)
            self.c_minus = np.mean(phi_X[y == -1], axis=0)
        
        # Initialize matrix A
        A = np.zeros((phi_X.shape[1], phi_X.shape[1]))

        for epoch in range(self.epochs):
            permutation = np.random.permutation(X.shape[0])
            phi_X_shuffled = phi_X[permutation]
            y_shuffled = y[permutation]
            
            for i in range(0, X.shape[0], self.batch_size):
                end = i + self.batch_size
                phi_X_batch = phi_X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                # Process each sample in the batch
                for j in range(phi_X_batch.shape[0]):
                    if y_batch[j] == 1:
                        A += (phi_X_batch[j] - self.c_plus).reshape(-1, 1) @ (phi_X_batch[j] - self.c_plus).reshape(1, -1)
                    else:
                        A += (phi_X_batch[j] - self.c_minus).reshape(-1, 1) @ (phi_X_batch[j] - self.c_minus).reshape(1, -1)
        
        # Rescale A
        A *= (2 * self.gamma / (X.shape[0] * self.epochs))
        
        # Solving for weights
        I = np.eye(phi_X.shape[1])
        self.weights = np.linalg.inv(I + A) @ (phi_X.T @ y)

    def predict(self, X):
        phi_X = np.hstack((X, np.ones((X.shape[0], 1))))  # φ(x) | 1
        return np.sign(phi_X @ self.weights)

psvm = PSVM(0.1,10,2)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_embeddings, tags, test_size=0.2, random_state=42)

# Train PSVM
psvm.fit(X_train, y_train)

# Predict and evaluate
y_pred = psvm.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Random key encoding
def random_key_encoding(population_size, hyperparameters):
    population = []
    for _ in range(population_size):
        individual = []
        for param, space in hyperparameters.items():
            if isinstance(space, np.ndarray):
                # For integer and continuous spaces
                individual.append(np.random.choice(space))
            else:
                # For categorical spaces
                individual.append(np.random.choice(range(len(space))))
        population.append(individual)
    return np.array(population)


def De(population, F=0.8):
    # Apply k-means clustering to divide the population
    kmeans = KMeans(n_clusters=max(2, int(len(population)/5))).fit(population)  # Ensure at least 2 clusters
    labels = kmeans.labels_

    # Find the cluster with the minimum mean objective function value
    unique_labels = np.unique(labels)
    cluster_scores = [np.mean([objective_function(population[i]) for i in range(len(population)) if labels[i] == label]) for label in unique_labels]
    win_region = unique_labels[np.argmin(cluster_scores)]
    win_indices = np.where(labels == win_region)[0]

    # Perform the mutation based on the winning region
    mutated_population = []
    for i in range(len(population)):
        if len(win_indices) >= 2:
            r1, r2 = population[np.random.choice(win_indices, 2, replace=False)]
        elif len(win_indices) == 1:
            # Only one individual in the winning cluster, use it as r1 and select another from the entire population
            r1 = population[win_indices[0]]
            r2 = population[np.random.choice(np.delete(np.arange(len(population)), win_indices[0]))]
        else:
            # No individuals in the winning cluster, fall back to random selection from the entire population
            r1, r2 = population[np.random.choice(len(population), 2, replace=False)]

        win = population[np.random.choice(win_indices)] if len(win_indices) > 0 else population[np.random.randint(len(population))]
        mutant = win + F * (r1 - r2)
        mutated_population.append(mutant)

    return np.array(mutated_population)

def objective_function(individual):
    return np.random.rand()

def main_optimization_loop(population_size, generations, hyperparameters, F=0.8):
    # Initialize population using random key encoding
    population = random_key_encoding(population_size, hyperparameters)

    # Main loop for generations
    for generation in range(generations):
        # Perform mutation using the improved DE mutation strategy
        mutated_population = De (population, F=F)

        # Evaluate the original and mutated populations
        fitness_original = np.array([objective_function(individual) for individual in population])
        fitness_mutated = np.array([objective_function(individual) for individual in mutated_population])

        # Selection: Create a new population
        new_population = []
        for i in range(population_size):
            if fitness_mutated[i] < fitness_original[i]:  # Assuming minimization
                new_population.append(mutated_population[i])
            else:
                new_population.append(population[i])

        population = np.array(new_population)

        # Here, you could add logging of best performance, etc.
        print(f"Generation {generation}: Best Fitness = {np.min(fitness_original)}")

    # Return the final population and its fitness
    return population, fitness_original
