import numpy as np


def crossover(parents, size):
    i = 1
    children = []

    for parent_1 in parents:
        for parent_2 in parents[i:]:
            crossover_prob = np.random.randint(101)

            if crossover_prob > 60 and size <= 28:
                cohort = np.random.randint(10)

                child_1 = np.concatenate([parent_1[:cohort], parent_2[cohort:]])
                child_2 = np.concatenate([parent_2[:cohort], parent_1[cohort:]])

                children.append(child_1)
                children.append(child_2)

                size += 2
            i += 1

    return children, size


def mutations(parents, children, size):
    for chromosome in parents:
        mutation_probability = np.random.randint(101)
        #print(mutation_probability)
        if size <= 29 and mutation_probability > 90:

            children.append([])
            for position in chromosome:
                mutate = np.random.randint(2)

                if mutate:
                    if position == 0:
                        children[size].append(1)
                    else:
                        children[size].append(0)
                else:
                    children[size].append(position)

            size += 1

        return children, size


def invert(parents, children, size):
    for chromosome in parents:
        reversal_prob = np.random.randint(101)

        if size <= 29 and reversal_prob > 90:
            children.append(chromosome)

            p1 = np.random.randint(7)
            p2 = np.random.randint(8)
            while p2 < p1:
                p2 = np.random.randint(8)

            x = (p2 - p1) // 2

            for i in range(x):
                aux = children[size][p1 + i]
                children[size][p1 + i] = children[size][p2 - i]
                children[size][p2 - i] = aux

            size += 1

    return children, size


def get_fitness(chromossomes):
    chromossomes_fitness = np.zeros(len(chromossomes))

    for i in range(len(chromossomes)):
        for j in range(1, 7):
            if chromossomes[i][j] == 0 and chromossomes[i][j + 1] == 1:
                chromossomes_fitness[i] += 1

    return chromossomes_fitness

def replace(parents, children, parents_fit, children_fit, size):

    new_population, new_population_fitness = parents, parents_fit
    parents_index, children_index = 0,0

    for i in range(len(parents)):
        #print(len(parents), len(children))
        #print(parents_index, children_index)
        if parents_index < len(parents) and children_index < size - 1:

            if parents_fit[parents_index] > children_fit[children_index]:
                if i >= len(new_population):
                    new_population.append(parents[parents_index])
                    new_population_fitness.append(parents_fit[parents_index])
                else:
                    new_population[i]=parents[parents_index]
                    new_population_fitness[i] = parents_fit[parents_index]
                parents_index += 1

            else:
                if i >= len(new_population):
                    new_population.append(children[children_index])
                    new_population_fitness.append(children_fit[children_index])
                else:
                    new_population[i] = children[children_index]
                    new_population_fitness[i] = children_fit[children_index]
                children_index += 1

        elif parents_index < len(parents) and children_index > size - 1:
                if i >= len(new_population):
                    new_population.append(parents[parents_index])
                    new_population_fitness.append(parents_fit[parents_index])
                else:
                    new_population[i]=parents[parents_index]
                    new_population_fitness[i] = parents_fit[parents_index]
                parents_index += 1

        else:
            new_population[i] = children[children_index]
            new_population_fitness[i] = children_fit[children_index]
            children_index += 1


    return new_population, new_population_fitness

def order_by_fitness(chromossomes, chromosome_fitness):
    ordered_pop, ordered_fit = [], []
    fit_values_ordered = -np.sort(-np.unique(chromosome_fitness))

    for fit_value in fit_values_ordered:
        for x in np.array(chromossomes)[np.where(chromosome_fitness == fit_value)]:
            ordered_pop.append(x)
            ordered_fit.append(fit_value)

    return ordered_pop, ordered_fit


# Gera população
population = np.random.randint(2, size=(10, 8))

#print('População inicial gerada: ')
#print(population)

#print('\nPopulação inicial adaptada')
fitness = get_fitness(population)

#print(fitness)

population, fitness = order_by_fitness(population, fitness)
#print('\nPopulação inicial na ordem decrescente de adaptação')
#print(population)

while fitness[0] != 4:
    descendants_size = 0
    descendants, descendants_size = crossover(population[:5], descendants_size)

    #print('\nCruzamento:')
    #print(descendants)

    descendants, descendants_size = mutations(population[:5], descendants, descendants_size)

    #print('\nMutações:')
    #print(descendants)

    descendants, descendants_size = invert(population[:5], descendants, descendants_size)

    #print('\nInversões:')
    #print(descendants)

    descendants_fitness = get_fitness(descendants)

    #print('Índice de adaptação de cada descendente')
    #print(descendants_fitness)

    descendants, descendants_fitness = order_by_fitness(descendants, descendants_fitness)
    #print('\nPopulação descendente na ordem decrescente de adaptação')
    #print(descendants)

    population, fitness = replace(population,descendants, fitness, descendants_fitness, descendants_size)
    population, fitness = order_by_fitness(population, fitness)
    #print('\nPopulação nova')
    #print(population)

    print(fitness)


