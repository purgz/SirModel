import matplotlib as plt
import numpy as np

class Agent:
    def __init__(self, state): 
        # State can be 'S' 'I' or 'R'
        self.state = state

    def printState(self):
        print(self.state)

    def setState(self,state):
        self.state = state



# Some initial values
populationSize = 1000
numSusceptible = populationSize - 1
numInfected = 1
numRecovering = 0

# Initialise a population of 1 infected and the rest susceptible.
population = np.array([Agent("S") for i in range(populationSize)])
population[populationSize-1].setState("I")
population[populationSize-1].printState

"""for i in range(len(population)):
    population[i].printState()
"""



# Main loop - add some stopping criteria - maybe number of infected stops changing?
while True:


    break



