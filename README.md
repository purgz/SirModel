# SirModel
SIR agent based simulation - game theory coursework university of birmingham


Optional command line arguments (11):
[-disease] {3 options, COVID-19, INFLUENZA, MEASLES} 
[-n] {provide integer number, for lattice provide square number}
[-beta ] {custom infection rate, float value [0,1]}
[-gamma] {custom recovery rate, float value [0,1]}
[-repeats] {number of repetions for averaging, integer}
[-lattice] {True, for lattice/grid topology, otherwise ignored}
[-E] {efficacy, provide a float value [0,1]}
[-P] {proportion vaccinated, provide float value [0,1]}
[-animate] {provide number of seconds between each lattice grid print}
[-timesteps] {integer number, time steps simulation run for}
[-quarantine] {float [0,1], probability of quarantining upon infection}
[-reinfection] {float [0,1], probability of switching from R to S}


Examples:
Grid printing:
python .\2422764_HenryBrooks.py -disease MEASLES -n 400 -lattice True -E 0.95 -P 0.95 -quarantine 0.4 -timesteps 100 -animate 1

Oscillating
python .\2422764_HenryBrooks.py -disease MEASLES -n 900 -lattice True -E 0.95 -P 0.95 -reinfection 0.05 -timesteps 300

Regular COVID-19
python .\2422764_HenryBrooks.py -disease COVID-19 -n 400  -E 0.95 -P 0.95

Lattice Influenza:
python .\2422764_HenryBrooks.py -disease INFLUENZA -n 900 -lattice True 