from itertools import combinations, groupby
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import warnings
from sklearn.exceptions import ConvergenceWarning

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

class GeneticOptimizedNNModel:
    def __init__(self, size):
        self.population = []
        self.size = size
        self.fitness = []
        self.hyperparameters = {
            'n_iter': 'int',
            'optimizer': 'cat',
            'activation': 'cat',
        }
        self.n_iter = [20, 30, 40, 50, 60, 70]
        self.optimizers = ["adam", "sgd", "lbfgs"]
        self.activations = ['identity', 'logistic', 'tanh', 'relu']

    def initPopulation(self):

        self.population = []

        while len(self.population) < self.size:
            child = {}
            for hyperparameterName in self.hyperparameters.keys():

                if hyperparameterName == 'n_iter':
                    child[hyperparameterName] = np.random.choice(self.n_iter)
                if hyperparameterName == 'optimizer':
                    child[hyperparameterName] = np.random.choice(
                        self.optimizers)
                if hyperparameterName == 'activation':
                    child[hyperparameterName] = np.random.choice(
                        self.activations)

            if not self.isMonster(child):
                self.population.append(child)

    def crossover(self, pCrossover):

        new_chromossomes = []

        for (a, b) in combinations(self.population, 2):

            if a != b:

                prob = np.random.randint(1, 100) / 100

                if prob <= pCrossover:

                    child1 = {}
                    child2 = {}

                    currentSize = len(new_chromossomes)

                    while currentSize == len(new_chromossomes):

                        for hyperparameterName in self.hyperparameters.keys():

                            if hyperparameterName == 'activation':
                                child1[hyperparameterName] = a[hyperparameterName]
                                child2[hyperparameterName] = b[hyperparameterName]

                            if hyperparameterName == 'optimizer':
                                child1[hyperparameterName] = b[hyperparameterName]
                                child2[hyperparameterName] = a[hyperparameterName]

                            if hyperparameterName == 'n_iter':
                                choice = np.random.choice([0,1])

                                if choice == 1:
                                    child1[hyperparameterName] = a[hyperparameterName]
                                    child2[hyperparameterName] = b[hyperparameterName]
                                else:
                                    child1[hyperparameterName] = b[hyperparameterName]
                                    child2[hyperparameterName] = a[hyperparameterName]


                            # if self.hyperparameters[hyperparameterName] == 'int':
                            #     weight = np.random.randint(1, 9) / 10
                            #     vb = b[hyperparameterName]
                            #     va = a[hyperparameterName]
                            #     child1[hyperparameterName] = round(
                            #         a[hyperparameterName] * weight + b[hyperparameterName] * (1-weight))
                            #     child2[hyperparameterName] = round(va * weight - vb * (1-weight) if va * weight > vb * (1-weight) else vb * weight - va * (1-weight))

                        if not self.isMonster(child1):
                            new_chromossomes.append(child1)
                        # print("CROSSOVER EFETUADO")

                        if not self.isMonster(child2):
                            new_chromossomes.append(child2)
                        # print("CROSSOVER EFETUADO")

        self.population = self.population + new_chromossomes

    def mutation(self, pMutation):

        new_chromossomes = []

        for parent in self.population:

            prob = np.random.randint(1, 100) / 100

            if prob <= pMutation:

                child = parent.copy()

                currentSize = len(new_chromossomes)

                while currentSize == len(new_chromossomes):

                    hyperparameterName = np.random.choice(
                        list(self.hyperparameters.keys()))

                    if hyperparameterName == 'n_iter':
                        child[hyperparameterName] = np.random.choice(self.n_iter)
                    if hyperparameterName == 'optimizer':
                        child[hyperparameterName] = np.random.choice(
                            self.optimizers)
                    if hyperparameterName == 'activation':
                        child[hyperparameterName] = np.random.choice(
                            self.activations)

                    if not self.isMonster(child):
                        new_chromossomes.append(child)
                       # print("MUTAÇÃO EFETUADA")

        self.population = self.population + new_chromossomes

    def select(self):
        sort = np.argsort(self.fitness)[::-1][:self.size]
        tmp = [self.population[i] for i in sort]
        self.population = tmp

        tmp = [self.fitness[i] for i in sort]
        self.fitness = tmp
        #print(self.population)

    def isMonster(self, chromossome):

        if chromossome["n_iter"] not in self.n_iter:
            return True
        if chromossome["optimizer"] not in self.optimizers:
            return True
        if chromossome["activation"] not in self.activations:
            return True

        return False

    # def reachMinimum(self, minimumScore):
    #     for item in self.fitness:
    #         if item >= minimumScore:
    #             return True

    #     return False

    def run(self, X_train, X_test, y_train, y_test, pCrossover, pMutation):

        self.initPopulation()
        # print("POPULAÇÃO INICIAL  ")
        # for i in self.population:
        #     print(i)
        iCounter = 1
        converged = False
        while not converged:

            self.crossover(pCrossover)
            # print("POPULAÇÃO APÓS CROSSOVER")
            # for i in self.population:
            #     print(i)

            self.mutation(pMutation)

            # print("POPULAÇÃO APÓS MUTAÇÃO")

            # for i in self.population:
            #     print(i)
            self.fitness = []
           
            for chromossome in self.population:
                
                # INICIO: CÓDIGO PARA IGNORAR WARNINGS
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                # FIM: CÓDIGO PARA IGNORAR WARNINGS
                    model = MLPClassifier(max_iter=chromossome['n_iter'], solver=chromossome['optimizer'], activation=chromossome['activation'])
                    model.fit(X_train, y_train) 
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test, y_pred)

                    self.fitness.append(score)

                    if score >= max(self.fitness):
                        best_model = {
                            'f1': f1_score(y_test, y_pred),
                            'acc': accuracy_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred)

                        }

            self.select()
            
            converged = all_equal(self.population)
            
            print(f" Iteração {iCounter} - POPULAÇÃO: {self.population}")
            iCounter += 1

        print("\n\nCONVERGÊNCIA ATINGIDA")
        print(f"Scores: {best_model}")
        print("\nMELHOR CROMOSSOMO")
        print(self.population[0])