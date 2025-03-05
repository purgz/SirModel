import matplotlib.pyplot as plt
import numpy as np
import random



"""
Game theory coursework code - Henry Brooks
Student id: 2422764
"""


# Basic agent class for simple model
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
#R0 = beta / gamma

# Parameters - I think these were estimates for covid
Rnaught = 2.79
recoveryRate = 1/14
infectionRate = recoveryRate * Rnaught

# Initialise a population of 1 infected and the rest susceptible.
# Population is represented as a simple list of agents
population = np.array([Agent("S") for i in range(populationSize)])
population[populationSize-1].setState("I")
population[populationSize-1].printState

population2 = np.array([Agent("S") for i in range(populationSize)])
population2[populationSize-1].setState("I")
population2[populationSize-1].printState


def updateProcess(agent1, agent2, infectionRate, numInfected, numSusceptible):
    # Infects with probability beta
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

    return numInfected, numSusceptible


# Would like to add more complex agents, where the infection rates are individual and based on vaccination / quarantining etc.

def runSimulation(population, infectionRate, recoveryRate, vaccinate):
    # Graph data
    infectedOverT = []
    susOverT = []
    recOverT = []

    populationSize = 1000
    numSusceptible = populationSize - 1
    numInfected = 1
    numRecovering = 0

    # Improve this model where indivudals are vaccinated in the agent class and this applies to their infection and recovery rates.
    # Assuming vaccine efficacy of 50 %
    if (vaccinate):
        infectionRate *= 0.5
    

    count = 0
    # Main loop - Stopping criteria when no more infected individuals
    while numInfected != 0 and count < 100:



        # Have inner loop of each individual
        if numSusceptible != 0: # for efficiency - all infected so no need to perform this loop
        
            for i in range(populationSize):

                # The selected individual meets a random individual (agentb) and then update process is applied.
                agent = population[i]
        
                r = np.random.choice([x for x in range(populationSize) if x != i])
                agentb = population[r]

                # Apply the update process for two individuals
                numInfected, numSusceptible = updateProcess(agent, agentb, infectionRate, numInfected, numSusceptible)

        # Check each agent recovers at this timestep
        for i in range(populationSize):
            recoverAgent = population[i]
            if recoverAgent.state == "I":
                if random.random() < recoveryRate:
                    recoverAgent.setState("R")
                    numRecovering += 1
                    numInfected -= 1


        if count % 1 == 0:
            print("********")
            print("Generation " , count)
            print("Infected ", numInfected)
            print("Susceptible ", numSusceptible)
            print("Recovering ", numRecovering)

        infectedOverT.append(numInfected)
        susOverT.append(numSusceptible)
        recOverT.append(numRecovering)

        count += 1
    
    return infectedOverT, susOverT, recOverT

infectedOverT, susOverT, recOverT = runSimulation(population, infectionRate, recoveryRate, vaccinate = False)

# Simple vaccination assumption
infectedOverTV, susOverTV, recOverTV = runSimulation(population2, infectionRate, recoveryRate, vaccinate = True)


fig, axes = plt.subplots(1, 2, figsize=(10,4))

axes[0].plot(range(len(infectedOverT)), infectedOverT, color='r', label='Infected over time')
axes[0].plot(range(len(infectedOverT)), susOverT, color='g', label='Susceptible over time')
axes[0].plot(range(len(infectedOverT)), recOverT, color='b', label='Recovered over time')
axes[0].set_title("SIR results for unvaccinated")
axes[0].legend()

# Second plot (right)
axes[1].plot(range(len(infectedOverTV)), infectedOverTV, color='r', label='Infected over time')
axes[1].plot(range(len(infectedOverTV)), susOverTV, color='g', label='Susceptible over time')
axes[1].plot(range(len(infectedOverTV)), recOverTV, color='b', label='Recovered over time')
axes[1].set_title("SIR results for vaccinated")
axes[1].legend()

# Show the plots
plt.tight_layout()
plt.show()