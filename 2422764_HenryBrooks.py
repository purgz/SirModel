import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import argparse
import ternary

"""
Libraries required
matplotlib
numpy
python-ternary
"""

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

parser.add_argument("-repeats", "--repeats")
parser.add_argument("-lattice", "--lattice")

args = parser.parse_args()

repeats = 1
if args.repeats != None:
    repeats = int(args.repeats)

useLattice = False

if args.lattice != None:
    if args.lattice == "True":
        useLattice = True


# Constants
COVID_R0 = 3.32
COVID_GAMMA = 1/10 # 10 day infectious period
COVID_BETA = COVID_R0 * COVID_GAMMA

FLU_R0 = 1.28
FLU_GAMMA = 1 / 5
FLU_BETA = FLU_R0 * FLU_GAMMA

MEASLES_R0 = 17.5
MEASLES_GAMMA = 1 / 4
MEASLES_BETA = MEASLES_GAMMA * MEASLES_R0

# Parameters - Values for COVID-19 from my report - default if no comand line args given
Rnaught = 3.32
recoveryRate = 1/10
infectionRate = recoveryRate * Rnaught


if args.disease != None:
    if args.disease == "COVID-19":
      infectionRate = COVID_BETA
      recoveryRate = COVID_GAMMA
    elif args.disease == "INFLUENZA":
        infectionRate = FLU_BETA
        recoveryRate = FLU_GAMMA
    elif args.disease == "MEASLES":
        infectionRate = MEASLES_BETA
        recoveryRate = MEASLES_GAMMA
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

populationSize = 900
if args.popsize != None:
  populationSize = int(args.popsize)

numSusceptible = populationSize - 1
numInfected = 1
numRecovering = 0

# Vaccine effectiveness - assumption
vaccineEffect = 0.96


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


def getNeighbours(lattice, x, y):
    neighbours = []
   
    dxdy = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]

    for dx,dy in dxdy:
        a,b = x + dx , y + dy

        if (0 <= a < len(lattice) and 0 <= b < len(lattice[0])):  
            neighbours.append(lattice[a][b])

    return neighbours

def latticeUpdateProcess(lattice, infectionRate, numInfected, numSusceptible):
   
   infectedCoords = [(x,y) for x in range(len(lattice)) for y in range(len(lattice[0])) if lattice[x][y].state == "I"]

   for x,y in infectedCoords:
      agent1 = lattice[x][y]

      neighbours = getNeighbours(lattice, x , y)

      for agent2 in neighbours:
         numInfected, numSusceptible = updateProcess(agent1, agent2, infectionRate, numInfected, numSusceptible)
         
   return lattice, numInfected, numSusceptible

# Would like to add more complex agents, where the infection rates are individual and based on vaccination / quarantining etc.

def runSimulation(population, infectionRate, recoveryRate, nLattice):


    # Ideally there should be a check here that populaton size is a perfect square iff nlattice is set to true.

    # Graph data
    infectedOverT = []
    susOverT = []
    recOverT = []

    #populationSize = 1000
    numSusceptible = populationSize - 1
    numInfected = 1
    numRecovering = 0

    # Extra data for lattice arrangement
    lattice = []
    rowsN = int(np.sqrt(len(population)))
    for i in range(rowsN):
       lattice.append(population[i*rowsN : (i+1)*rowsN])

    
    if nLattice:
        print("INITIAL LATTICE")
        for row in lattice:
            print(" | ".join(f"{agent.state:2}" for agent in row))

    count = 0
    # Main loop - Stopping criteria when no more infected individuals
    while count < 100: #numInfected != 0 and 

        if nLattice:

            print("******************************************")
            print(count, numInfected, numSusceptible)
            for row in lattice:
                print(" | ".join(f"{agent.state:2}" for agent in row))
            lattice, numInfected, numSusceptible = latticeUpdateProcess(lattice, infectionRate, numInfected, numSusceptible)

        else:
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


        #
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


# Default interaction topology is all to all
# 2nd topology is NxN lattice. - required population size to be square number for convenience...
# Introducing the lattice I suppose will only require changing the update process.


infectedOverT, susOverT, recOverT = [],[],[]
infectedOverTV, susOverTV, recOverTV = [],[],[]
#Run simulation n times
for i in range(repeats):
    population = copy.deepcopy(INITIAL_POPULATION)
    infectedOverTi, susOverTi, recOverTi = runSimulation(population, infectionRate, recoveryRate, nLattice = useLattice)

    population2 = copy.deepcopy(INITIAL_POPULATION)
    for i, ind in enumerate(population2):
      if random.random() < 0.65:
        ind.vaccinate()
    infectedOverTVi, susOverTVi, recOverTVi = runSimulation(population2, infectionRate, recoveryRate, nLattice = useLattice)

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


# Trajectory
total = infectedOverT + susOverT + recOverT
infNorm = infectedOverT / total
susNorm = susOverT / total
recNorm = recOverT / total

totalV = infectedOverTV + susOverTV + recOverTV
infNormV = infectedOverTV / totalV
susNormV = susOverTV / totalV
recNormV = recOverTV / totalV

print(susOverTV[0] / totalV)


traj = list(zip(susNorm, infNorm, recNorm))
trajV = list(zip(susNormV, infNormV, recNormV))

fig, axes = plt.subplots(2 ,2 ,figsize=(12,6))


tax = ternary.TernaryAxesSubplot(ax=axes[0,1],scale=1.0)
tax.boundary()
tax.gridlines(color="gray", multiple=0.1)
tax.plot(traj, linewidth=2, label="SIR dynamics Unvaccinated", marker='o')
tax.right_corner_label("S ", fontsize=12)
tax.top_corner_label("I", fontsize=12)
tax.left_corner_label("R", fontsize=12)
tax.legend()

tax2 = ternary.TernaryAxesSubplot(ax=axes[1,1],scale=1.0)
tax2.boundary()
tax2.gridlines(color="gray", multiple=0.1)
tax2.plot(trajV, linewidth=2, label="SIR dynamics Vaccinated", marker='o')
tax2.right_corner_label("S", fontsize=12)
tax2.top_corner_label("I", fontsize=12)
tax2.left_corner_label("R", fontsize=12)
tax2.legend()






#fig, axes = plt.subplots(1, 2, figsize=(10,4))

axes[0,0].plot(range(len(infectedOverT)), infectedOverT, color='r', label='Infected over time')
axes[0,0].plot(range(len(infectedOverT)), susOverT, color='g', label='Susceptible over time')
axes[0,0].plot(range(len(infectedOverT)), recOverT, color='b', label='Recovered over time')
axes[0,0].set_title("SIR results for unvaccinated")
axes[0,0].legend()

# Second plot (right)
axes[1,0].plot(range(len(infectedOverTV)), infectedOverTV, color='r', label='Infected over time')
axes[1,0].plot(range(len(infectedOverTV)), susOverTV, color='g', label='Susceptible over time')
axes[1,0].plot(range(len(infectedOverTV)), recOverTV, color='b', label='Recovered over time')
axes[1,0].set_title("SIR results for vaccinated")
axes[1,0].legend()

# Show the plots
plt.tight_layout()
plt.show()



## Todo - improve vaccination using individual agent vaccination rates

