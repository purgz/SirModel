import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import argparse

"""
Game theory coursework code - Henry Brooks
Student id: 2422764
"""

"""
Command line arguments:
-disease, some predefined choices of disease with beta and gamma defined, options = {COVID-19}
-beta, override beta  value with float between 0 and 1
-gamma, override gamma value with float between 0 and 1
-n, give custom population size
"""

parser = argparse.ArgumentParser()
parser.add_argument("-disease", "--disease", help="Enter disease, options = COVID-19,...")
parser.add_argument("-n", "--popsize", help="Enter population size")


# Arguments for custom beta and gamma values - takes precidence over -disease
parser.add_argument("-beta", "--beta", help="Enter custom beta value")
parser.add_argument("-gamma", "--gamma", help="Enter custom gamma argument")

args = parser.parse_args()

# Constants
COVID_R0 = 3.32
COVID_GAMMA = 1/10 # 10 day infectious period
COVID_BETA = COVID_R0 * COVID_GAMMA

# Parameters - Values for COVID-19 from my report - default if no comand line args given
Rnaught = 3.32
recoveryRate = 1/10
infectionRate = recoveryRate * Rnaught


if args.disease != None:
    if args.disease == "COVID-19":
      infectionRate = COVID_BETA
      recoveryRate = COVID_GAMMA
    else:
      print("INVALID DISEASE NAME")
      exit()

    print("Disease ", args.disease)

if args.beta != None:
  print("Override beta ", args.beta)
  infectionRate = float(args.beta)
if args.gamma != None:
  print("Override gamma ", args.gamma)
  recoveryRate = float(args.gamma)


print("INFECTION RATE ", infectionRate)
print("RECOVERY RATE ", recoveryRate)


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
if args.popsize != None:
  populationSize = int(args.popsize)

numSusceptible = populationSize - 1
numInfected = 1
numRecovering = 0

# Vaccine effectiveness - assumption
vaccineEffect = 0.9


# Initialise a population of 1 infected and the rest susceptible.
# Population is represented as a simple list of agents
population = np.array([Agent("S", False) for i in range(populationSize)])
population[populationSize-1].setState("I")
population[populationSize-1].printState

INITIAL_POPULATION = copy.deepcopy(population)

population2 = copy.deepcopy(INITIAL_POPULATION)


def updateProcess(agent1, agent2, infectionRate, numInfected, numSusceptible):
    # Infects with probability beta
    # Calculate updated infection rate if the agent is vaccinated
    Pinf = infectionRate
    if (agent1.state == "I" and agent2.state == "S"):

        if (agent2.vaccinated):
          Pinf = (1 - vaccineEffect) * Pinf
        if random.random() < Pinf:
            agent2.setState("I")
            numInfected += 1
            numSusceptible -= 1
    elif (agent1.state == "S" and agent2.state == "I"):
        if (agent1.vaccinated):
          Pinf = (1 - vaccineEffect) * Pinf
        if random.random() < Pinf:
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

    #populationSize = 1000
    numSusceptible = populationSize - 1
    numInfected = 1
    numRecovering = 0

    # Improve this model where indivudals are vaccinated in the agent class and this applies to their infection and recovery rates.
    # Assuming vaccine efficacy of 50 %
    """if (vaccinate):
        infectionRate *= 0.5
    """

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


        if count % 20 == 0:
            print("********")
            print("Timestep " , count)
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
for i in range(1):
    population = copy.deepcopy(INITIAL_POPULATION)
    infectedOverTi, susOverTi, recOverTi = runSimulation(population, infectionRate, recoveryRate, vaccinate=False)

    population2 = copy.deepcopy(INITIAL_POPULATION)
    for i, ind in enumerate(population2):
      if random.random() < 0.65:
        ind.vaccinate()
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

