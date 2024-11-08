import random
import numpy as np
from FuncConsTransTest import main

"""Place the name of the file you want to run here"""
file = "FuncConsTransTest.py"

"""How many iterations you want to run here"""
iterations = 10

"""How many epochs per iteration here"""
epochs = 4

'''Do you want to randomize the seed value?'''
random_seed = True

fullAcc = []
for _ in range(iterations):
    if random_seed:
        seed = random.randint(1, 1000)
    else: seed = 42
    accuracies = main(seed, epochs, False, True)
    print("Seed: {seed}")
    for i, acc in enumerate(accuracies):
        print(f"Epoch {i}: {acc}")
        fullAcc.append(acc)

fullAcc = np.array(fullAcc)
print(f"Average accuracy across all runs and epochs: {fullAcc.mean()}")
