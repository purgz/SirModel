import matplotlib.pyplot as plt
import numpy as np
import random
import copy


"""
Game theory coursework code - Henry Brooks
Student id: 2422764
"""


# Basic agent class for simple model
class Agent:
    def __init__(self, state, vaccinated): 
        # State can be 'S' 'I' or 'R'
        self.state = state
        self.vaccinated = vaccinated

    def printState(self):
        print(self.state)

    def setState(self,state):
        self.state = state

    def vaccinate(self):
        self.vaccinated = True

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
population = np.array([Agent("S", False) for i in range(populationSize)])
population[populationSize-1].setState("I")
population[populationSize-1].printState

INITIAL_POPULATION = copy.deepcopy(population)

population2 = copy.deepcopy(INITIAL_POPULATION)


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
    while count < 100: #numInfected != 0 and 

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
        if (count > 7):   ## Arbitrary value to prevent single infected person recoverying on the first day
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



infectedOverT, susOverT, recOverT = [],[],[]
infectedOverTV, susOverTV, recOverTV = [],[],[]
#Run simulation n times
for i in range(5):
    population = copy.deepcopy(INITIAL_POPULATION)
    infectedOverTi, susOverTi, recOverTi = runSimulation(population, infectionRate, recoveryRate, vaccinate=False)

    population2 = copy.deepcopy(INITIAL_POPULATION)
    infectedOverTVi, susOverTVi, recOverTVi = runSimulation(population2, infectionRate, recoveryRate, vaccinate = True)

    infectedOverT.append(infectedOverTi)
    susOverT.append(susOverTi)
    recOverT.append(recOverTi)

    infectedOverTV.append(infectedOverTVi)
    susOverTV.append(susOverTVi)
    recOverTV.append(recOverTVi)

infectedOverT = np.mean(np.array(infectedOverT), axis=0)
susOverT = np.mean(np.array(susOverT), axis=0)
recOverT = np.mean(np.array(recOverT), axis=0)


infectedOverTV = np.mean(np.array(infectedOverTV), axis=0)
susOverTV = np.mean(np.array(susOverTV), axis=0)
recOverTV = np.mean(np.array(recOverTV), axis=0)



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



## Todo - improve vaccination using individual agent vaccination rates