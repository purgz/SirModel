import matplotlib.pyplot as plt
import numpy as np
import random

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


# Parameters
infectionRate = 0.4
recoveryRate = 0.3

# Initialise a population of 1 infected and the rest susceptible.
population = np.array([Agent("S") for i in range(populationSize)])
population[populationSize-1].setState("I")
population[populationSize-1].printState

"""for i in range(len(population)):
    population[i].printState()
"""

# Data for graphs
infectedOverT = []
susOverT = []
recOverT = []

count = 0
# Main loop - add some stopping criteria - maybe number of infected stops changing?
while numInfected != 0:
    
    # Select two random individuals for infection
    rand1, rand2, rand3 = np.random.choice(populationSize, 3, replace=False)

    agent1 = population[rand1]
    agent2 = population[rand2]

    if (agent1.state == "I" and agent2.state == "S"):
        if random.random() < infectionRate:
            agent2.setState("I")
            numInfected += 1
            numSusceptible -= 1
    elif (agent1.state == "S" and agent2.state == "I"):
        if random.random() < infectionRate:
            agent1.setState("I")
            numInfected += 1
            numSusceptible -= 1

    recoverAgent = population[rand3]
    if recoverAgent.state == "I":
        if random.random() < recoveryRate:
            recoverAgent.setState("R")
            numRecovering += 1
            numInfected -= 1

    if count % 100 == 0:
        print("********")
        print("Generation " , count)
        print("Infected ", numInfected)
        print("Susceptible ", numSusceptible)
        print("Recovering ", numRecovering)

    infectedOverT.append(numInfected)
    susOverT.append(numSusceptible)
    recOverT.append(numRecovering)

    count += 1


# Sample data: values over time
time = np.arange(10)  # Time steps 0 to 9
values = np.random.randint(1, 100, size=10)  # Random values

# Plot the data
plt.plot(range(len(infectedOverT)), infectedOverT, label="Infected")

plt.plot(range(len(infectedOverT)), susOverT, label="Susceptible")

plt.plot(range(len(infectedOverT)), recOverT, label="Recovering")

# Labels and title
plt.xlabel("Time")
plt.ylabel("Number of individuals")
plt.title("Susceptible, infected, and recovering over time")
plt.legend()

# Show the graph
plt.show()
