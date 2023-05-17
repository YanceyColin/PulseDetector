####### Low Detector V3 #######
#                             #
#      By: Colin Yancey       #
#      Created Jan. 2023      #
#   Last Edited: 05/10/2023   #
#        Schulman Lab         #
#  Johns Hopkins University   #
#                             #
###############################

from pathlib import Path
import sys
sys.path.insert(1, str(Path('../')))
import numpy as np

import GeneletSystemModel as GSM
import ArrayPlots as AP

newGN = GSM.GeneletNetwork(
    #                        5                             10                             15
    ["G2", "G2", "G7", "G7", "G8", "G3", "G1", "G1", "G5", "G6", "G9", "G9", "G9",  "G4", "G10", "G11"],
    ["C1", "C3", "R4", "C8", "C9", "R1", "C5", "C6", "C4","CS4", "R5", "R6","RS4","RS10", "R11", "R10"]
)
newGN.setInitialConditions(
    #                            5                                  10                                 15
    ["OFF", "OFF", "OFF", "OFF", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "OFF", "ON"],
    [   50,    25,    10,    25,    30,    50,    50,    20,    10,    20,    50,    50,    50,    10,     5,    5],
    GSM.createGeneralClassProperties(
        #                   5                         10
        [250, 250, 125, 50, 100, 150, 250, 125,  400, 200, 200],
        [  0,   0,   0, 20, 190, 170,   0,   0, 2000,   0,   0]
    )
)

tM = 9 # Final time for bandpass
simTime = 16
simTimeVals = GSM.getTimeValues(simTime)
rtVal = simTime/1000*3600*0.02 # Convert the active concentrations to total RNA produced by accounting for time and rate constant vs. interpolation


numOfSims = 101 # Minimum 2, start and end point of profile
maxActivator = 330 # Use A1 concentrations between 0 nM and this value, in nM

inputVals = np.zeros(numOfSims)
bandpassProfile = np.zeros(numOfSims)
C4Prod = np.zeros(numOfSims)
CS4Prod = np.zeros(numOfSims)
RS10Prod = np.zeros(numOfSims)

measureCutoff = int(1000*11/simTime) # Time where we assume the steady state has been reached, we will take a summation up to here
t0TOtM = int(1000*tM/simTime)

for i in np.arange(numOfSims):
    newGN.modifyValue("Activator", 0, maxActivator*i/(numOfSims-1))
    newGN.simulate(simTimeVals)
    inputVals[i] = sum(newGN.OutputConcentrations["Act: G1->C5"])*rtVal
    bandpassProfile[i] = newGN.OutputConcentrations["Act: G10->R11"][measureCutoff]
    C4Prod[i] = sum(newGN.OutputConcentrations["Act: G5->C4"])*rtVal
    CS4Prod[i] = sum(newGN.OutputConcentrations["Act: G6->CS4"])*rtVal
    RS10Prod[i] = sum(newGN.OutputConcentrations["Act: G4->RS10"][:measureCutoff])*rtVal
    if i==int(.5*numOfSims):
        c4Curve = newGN.OutputConcentrations["Act: G5->C4"]
        cs4Curve = newGN.OutputConcentrations["Act: G6->CS4"]


c4Ic4Curve = [
    [
        [
            []
        ],
        [
            []
        ]
    ],
    [
        [
            [
                c4Curve[:t0TOtM],
                cs4Curve[:t0TOtM]
            ]
        ],
        [
            []
        ]
    ]
]

AP.generatePlots(simTimeVals[:t0TOtM]/3600, c4Ic4Curve,
    localXTitle="Time (hrs)",
    localYTitle="Concentration (nM)",
    specialOpts = [
        ["Legend", [[], [["ON G5C4", "ON G6CS4"]]]]
    ],
    colorOffset = 0.5,
    fixedY=(-.3, 10.3),
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_5_B_1"
)


C4ProdAndCS4Prod = [
    [
        [[]],
        [[]]
    ],
    [
        [
            [
                C4Prod,
                CS4Prod
            ]
        ],
        [[]]
    ]
]

AP.generatePlots(inputVals/max(inputVals), C4ProdAndCS4Prod,
    localXTitle="Input Pulse Size (nM)",
    localYTitle="Total Activity (nM RNA Output)",
    specialOpts = [
        ["Legend", [[], [["G5C4", "G6CS4"]]]]
    ],
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_5_C_1"
)


subtractionC4AndCS4 = [
    [
        [[]],
        [[]]
    ],
    [
        [
            [
                C4Prod - CS4Prod,
                []
            ]
        ],
        [[]]
    ]
]

AP.generatePlots(inputVals/max(inputVals), subtractionC4AndCS4,
    localXTitle="Input Pulse Size (nM)",
    localYTitle="Total Activity (nM RNA Output)",
    specialOpts = [
        ["Legend", [[["G5C4 - G6CS4"]]]]
    ],
    colorOffset = 1.5,
    fixedY=(-25, 440),
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_5_CD_2"
)


bandpassOutput = [
    [
        [[]],
        [[]]
    ],
    [
        [
            [
                [],
                RS10Prod
            ]
        ],
        [[]]
    ]
]

AP.generatePlots(inputVals/max(inputVals), bandpassOutput,
    localXTitle="Input Pulse Size (nM)",
    localYTitle="Total Activity (nM RNA Output)",
    specialOpts = [
        ["Legend", [[["G4RS10"]]]]
    ],
    colorOffset = 1.5,
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_5_D_1"
)

# newGN.plot1Variable("Activator", 0, [30*i for i in range(12)], GSM.getTimeValues(16), [["Act: G1->C5"], ["Act: G4->RS10"], ["Act: G10->R11", "Act: G11->R10"]],
#     plotFileName="P:/PhD Work/Assets/GSM Images/Pulse Detector Analysis/DetectorV3Switches"
# )
