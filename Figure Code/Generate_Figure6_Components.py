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

newGNLow = GSM.GeneletNetwork(
    #                        5                             10                             15
    ["G2", "G2", "G7", "G7", "G8", "G3", "G1", "G1", "G5", "G6", "G9", "G9", "G9",  "G4", "G10", "G11"],
    ["C1", "C3", "R4", "C8", "C9", "R1", "C5", "C6", "C4","CS4", "R5", "R6","RS4","RS10", "R11", "R10"]
)
newGNLow.setInitialConditions(
    #                            5                                  10                                 15
    ["OFF", "OFF", "OFF", "OFF", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "OFF", "ON"],
    [   50,    25,    10,    25,    30,    50,    50,    20,    10,    20,    50,    50,    50,    10,     5,    5],
    GSM.createGeneralClassProperties(
        #                   5                         10
        [250, 250, 125, 50, 100, 150, 250, 125,  400, 200, 200],
        [  0,   0,   0, 20,  70, 100,   0,   0, 2000,   0,   0]
    )
)

newGNMed = GSM.GeneletNetwork(
    #                        5                             10                             15
    ["G2", "G2", "G7", "G7", "G8", "G3", "G1", "G1", "G5", "G6", "G9", "G9", "G9",  "G4", "G10", "G11"],
    ["C1", "C3", "R4", "C8", "C9", "R1", "C5", "C6", "C4","CS4", "R5", "R6","RS4","RS10", "R11", "R10"]
)
newGNMed.setInitialConditions(
    #                            5                                  10                                 15
    ["OFF", "OFF", "OFF", "OFF", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "OFF", "ON"],
    [   50,    25,    10,    25,    30,    50,    50,    20,    10,    20,    50,    50,    50,    10,     5,    5],
    GSM.createGeneralClassProperties(
        #                   5                         10
        [250, 250, 125, 50, 100, 150, 250, 125,  400, 200, 200],
        [  0,   0,   0, 20, 190, 170,   0,   0, 2000,   0,   0]
    )
)

newGNHigh = GSM.GeneletNetwork(
    #                        5                             10                             15
    ["G2", "G2", "G7", "G7", "G8", "G3", "G1", "G1", "G5", "G6", "G9", "G9", "G9",  "G4", "G10", "G11"],
    ["C1", "C3", "R4", "C8", "C9", "R1", "C5", "C6", "C4","CS4", "R5", "R6","RS4","RS10", "R11", "R10"]
)
newGNHigh.setInitialConditions(
    #                            5                                  10                                 15
    ["OFF", "OFF", "OFF", "OFF", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "BLK", "OFF", "ON"],
    [   50,    25,    10,    25,    30,    50,    50,    20,    10,    20,    50,    50,    50,    10,     5,    5],
    GSM.createGeneralClassProperties(
        #                   5                         10
        [250, 250, 125, 50, 100, 150, 250, 125,  400, 200, 200],
        [  0,   0,   0, 20, 320, 250,   0,   0, 2000,   0,   0]
    )
)


simTime = 16
simTimeVals = GSM.getTimeValues(simTime)
rtVal = simTime/1000*3600*0.02 # Convert the active concentrations to total RNA produced by accounting for time and rate constant vs. interpolation


numOfSims = 101 # Minimum 2, start and end point of profile
maxActivator = 330 # Use A1 concentrations between 0 nM and this value, in nM

inputVals = np.zeros(numOfSims)

bandpassProfileLow = np.zeros(numOfSims)
diffLow = np.zeros(numOfSims)
RS10ProdLow = np.zeros(numOfSims)

bandpassProfileMed = np.zeros(numOfSims)
diffMed = np.zeros(numOfSims)
RS10ProdMed = np.zeros(numOfSims)

bandpassProfileHigh = np.zeros(numOfSims)
diffHigh = np.zeros(numOfSims)
RS10ProdHigh = np.zeros(numOfSims)

measureCutoff = int(1000*11/simTime) # Time where we assume the steady state has been reached, we will take a summation up to here

for i in np.arange(numOfSims):
    newGNLow.modifyValue("Activator", 0, maxActivator*i/(numOfSims-1))
    newGNLow.simulate(simTimeVals)

    newGNMed.modifyValue("Activator", 0, maxActivator*i/(numOfSims-1))
    newGNMed.simulate(simTimeVals)

    newGNHigh.modifyValue("Activator", 0, maxActivator*i/(numOfSims-1))
    newGNHigh.simulate(simTimeVals)

    inputVals[i] = sum(newGNHigh.OutputConcentrations["Act: G1->C5"])*rtVal

    bandpassProfileLow[i] = newGNLow.OutputConcentrations["Act: G10->R11"][measureCutoff]
    diffLow[i] = sum(newGNLow.OutputConcentrations["Act: G5->C4"])*rtVal - sum(newGNLow.OutputConcentrations["Act: G6->CS4"])*rtVal
    RS10ProdLow[i] = sum(newGNLow.OutputConcentrations["Act: G4->RS10"][:measureCutoff])*rtVal

    bandpassProfileMed[i] = newGNMed.OutputConcentrations["Act: G10->R11"][measureCutoff]
    diffMed[i] = sum(newGNMed.OutputConcentrations["Act: G5->C4"])*rtVal - sum(newGNMed.OutputConcentrations["Act: G6->CS4"])*rtVal
    RS10ProdMed[i] = sum(newGNMed.OutputConcentrations["Act: G4->RS10"][:measureCutoff])*rtVal
    
    bandpassProfileHigh[i] = newGNHigh.OutputConcentrations["Act: G10->R11"][measureCutoff]
    diffHigh[i] = sum(newGNHigh.OutputConcentrations["Act: G5->C4"])*rtVal - sum(newGNHigh.OutputConcentrations["Act: G6->CS4"])*rtVal
    RS10ProdHigh[i] = sum(newGNHigh.OutputConcentrations["Act: G4->RS10"][:measureCutoff])*rtVal

totalG10R11 = 5 # Total G10R11 genelet (nM)

allEndpoints = [
    [
        [],
        []
    ],
    [
        [
            [
                bandpassProfileLow/totalG10R11,
                bandpassProfileMed/totalG10R11,
                bandpassProfileHigh/totalG10R11
            ]
        ],
        []
    ]
]

allDifferences = [
    [
        [],
        []
    ],
    [
        [
            [
                diffLow,
                diffMed,
                diffHigh
            ]
        ],
        []
    ]
]

allRS10Prod = [
    [
        [],
        []
    ],
    [
        [
            [
                RS10ProdLow,
                RS10ProdMed,
                RS10ProdHigh
            ]
        ],
        []
    ]
]

lowEndPt = [
    [
        [],
        [],
        []
    ],
    [
        [
            [
                bandpassProfileLow/totalG10R11,
                [],
                []
            ]
        ],
        [],
        []
    ]
]

medEndPt = [
    [
        [],
        [],
        []
    ],
    [
        [
            [
                [],
                bandpassProfileMed/totalG10R11,
                []
            ]
        ],
        [],
        []
    ]
]

highEndPt = [
    [
        [],
        [],
        []
    ],
    [
        [
            [
                [],
                [],
                bandpassProfileHigh/totalG10R11
            ]
        ],
        [],
        []
    ]
]

lowDiff = [
    [
        [],
        [],
        []
    ],
    [
        [
            [
                diffLow,
                [],
                []
            ]
        ],
        [],
        []
    ]
]

medDiff = [
    [
        [],
        [],
        []
    ],
    [
        [
            [
                [],
                diffMed,
                []
            ]
        ],
        [],
        []
    ]
]

highDiff = [
    [
        [],
        [],
        []
    ],
    [
        [
            [
                [],
                [],
                diffHigh
            ]
        ],
        [],
        []
    ]
]

lowRS10Prod = [
    [
        [],
        [],
        []
    ],
    [
        [
            [
                RS10ProdLow,
                [],
                []
            ]
        ],
        [],
        []
    ]
]

medRS10Prod = [
    [
        [],
        [],
        []
    ],
    [
        [
            [
                [],
                RS10ProdMed,
                []
            ]
        ],
        [],
        []
    ]
]

highRS10Prod = [
    [
        [],
        [],
        []
    ],
    [
        [
            [
                [],
                [],
                RS10ProdHigh
            ]
        ],
        [],
        []
    ]
]

AP.generatePlots(inputVals/max(inputVals), allEndpoints,
    localXTitle="Input Pulse Size",
    localYTitle="Fraction ON G10R11",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_A_1",
    fixedY=(-0.04, 1.0)
)

AP.generatePlots(inputVals/max(inputVals), allDifferences,
    localXTitle="Input Pulse Size",
    localYTitle="Total Activity G5C4 - G5CS4 (nM)",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_B_1",
    colorOffset=1.5,
    fixedY=(-30, 570)
)

AP.generatePlots(inputVals/max(inputVals), allRS10Prod,
    localXTitle="Input Pulse Size",
    localYTitle="Total Activity G4RS10 (nM)",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_NA_1",
    colorOffset=2,
    fixedY=(-90, 2100)
)



AP.generatePlots(inputVals/max(inputVals), lowEndPt,
    localXTitle="Input Pulse Size",
    localYTitle="Fraction ON G10R11 @ 11 HR",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_C_1",
    fixedY=(-0.04, 1.0)
)

AP.generatePlots(inputVals/max(inputVals), medEndPt,
    localXTitle="Input Pulse Size",
    localYTitle="Fraction ON G10R11 @ 11 HR",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_D_1",
    fixedY=(-0.04, 1.0)
)

AP.generatePlots(inputVals/max(inputVals), highEndPt,
    localXTitle="Input Pulse Size",
    localYTitle="Fraction ON G10R11 @ 11 HR",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_E_1",
    fixedY=(-0.04, 1.0)
)



AP.generatePlots(inputVals/max(inputVals), lowDiff,
    localXTitle="Input Pulse Size",
    localYTitle="Total Activity G5C4 - G5CS4 (nM)",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_C_2",
    colorOffset=1.5,
    fixedY=(-30, 570)
)

AP.generatePlots(inputVals/max(inputVals), medDiff,
    localXTitle="Input Pulse Size",
    localYTitle="Total Activity G5C4 - G5CS4 (nM)",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_D_2",
    colorOffset=1.5,
    fixedY=(-30, 570)
)

AP.generatePlots(inputVals/max(inputVals), highDiff,
    localXTitle="Input Pulse Size",
    localYTitle="Total Activity G5C4 - G5CS4 (nM)",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_E_2",
    colorOffset=1.5,
    fixedY=(-30, 570)
)



AP.generatePlots(inputVals/max(inputVals), lowRS10Prod,
    localXTitle="Input Pulse Size",
    localYTitle="Total Activity G4RS10 (nM)",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_C_3",
    colorOffset=2,
    fixedY=(-90, 2100)
)

AP.generatePlots(inputVals/max(inputVals), medRS10Prod,
    localXTitle="Input Pulse Size",
    localYTitle="Total Activity G4RS10 (nM)",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_D_3",
    colorOffset=2,
    fixedY=(-90, 2100)
)

AP.generatePlots(inputVals/max(inputVals), highRS10Prod,
    localXTitle="Input Pulse Size",
    localYTitle="Total Activity G4RS10 (nM)",
    plotFileName="P:/PhD Work/Assets/GSM Images/FigurePlots/Figure_6_E_3",
    colorOffset=2,
    fixedY=(-90, 2100)
)
