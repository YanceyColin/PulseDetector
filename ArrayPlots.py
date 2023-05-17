#### GSM-Specific Plotter #####
#                             #
#      By: Colin Yancey       #
#      Created Dec. 2021      #
#   Last Edited: 2/16/2023    #
#        Schulman Lab         #
#  Johns Hopkins University   #
#                             #
###############################

import numpy as np
import matplotlib.pyplot as plt

import GeneralUse as GU


# Translates a particular hue value and brightness value into a normalized vector3 representation of the color
def getVector3FromHueAndBrightness(hueVal, brightnessVal):
    if hueVal>3 or hueVal<0 or brightnessVal>1 or brightnessVal<0: # hueVal is 0 - 3, shadeVal is 0 - 1
        raise ValueError("The given hue must be between 0.0 - 3.0, and the given shade must be between 0.0 - 1.0")

    color3 = np.array([ # Create the vector-3
        1-GU.bufferNumber(hueVal)+GU.bufferNumber(hueVal-1.5),
        GU.bufferNumber(hueVal+.5)-GU.bufferNumber(hueVal-1)+GU.bufferNumber(hueVal-2.5),
        GU.bufferNumber(hueVal-.5)-GU.bufferNumber(hueVal-2)
    ])

    if brightnessVal<.5: # Increase or decrease the brightness of the given hue
        color3 = 2*brightnessVal*color3
    elif brightnessVal>.5:
        color3 = (2-2*brightnessVal)*color3 + (2*brightnessVal-1)*np.array([1,1,1])

    return tuple(color3)

# Generates a list of hues maximally apart
def generateHueList(extraHues, offset=0): # Hues are values of 0 - 3 so offset should be input with that context in mind (an offset of 3 is essentially an offset of 0)
    return (3/extraHues*np.arange(extraHues)+offset)%3

# Creates a hue array for an array plot to use
def makeHueArray(hueInfo, iLen, jLen, unique=True, offset=0):
    iRange, jRange = range(iLen), range(jLen)
    if hueInfo["Type"]=="Uniform": # Unique is obsolete for uniform since all cells
        hueTab = generateHueList(hueInfo["GroupData"], offset=offset)
        return [[hueTab for _ in jRange] for _ in iRange]
    elif hueInfo["Type"]=="RowBased":
        if unique:
            masterHues = generateHueList(sum(hueInfo["GroupData"]), offset=offset)
            hueTab = [[[] for _ in jRange] for _ in iRange]
            priorNum = 0
            for i in iRange:
                for j in jRange:
                    hueTab[i][j] = masterHues[priorNum:(priorNum+hueInfo["GroupData"][i])]
                priorNum += hueInfo["GroupData"][i]
            return hueTab
        else:
            masterHues = generateHueList(max(hueInfo["GroupData"]), offset=offset)
            return [[masterHues[:hueInfo["GroupData"][i]] for j in jRange] for i in iRange]
    elif hueInfo["Type"]=="ColumnBased":
        if unique:
            masterHues = generateHueList(sum(hueInfo["GroupData"]), offset=offset)
            hueTab = [[[] for _ in jRange] for _ in iRange]
            priorNum = 0
            for j in jRange:
                for i in iRange:
                    hueTab[i][j] = masterHues[priorNum:(priorNum+hueInfo["GroupData"][j])]
                priorNum += hueInfo["GroupData"][j]
            return hueTab
        else:
            masterHues = generateHueList(max(hueInfo["GroupData"]), offset=offset)
            return [[masterHues[:hueInfo["GroupData"][j]] for j in jRange] for i in iRange]
    elif hueInfo["Type"]=="CellBased":
        if unique:
            masterHues = generateHueList(sum([sum(x) for x in hueInfo["GroupData"]]), offset=offset)
            hueTab = [[[] for _ in jRange] for _ in iRange]
            priorNum = 0
            for i in iRange:
                for j in jRange:
                    hueTab[i][j] = masterHues[priorNum:(priorNum+hueInfo["GroupData"][i][j])]
                    priorNum += hueInfo["GroupData"][i][j]
            return hueTab
        else:
            masterHues = generateHueList(max([max(x) for x in hueInfo["GroupData"]]), offset=offset)
            return [[masterHues[:hueInfo["GroupData"][i][j]] for j in jRange] for i in iRange]
    else:
        raise ValueError("Invalid hue type given; info must be Uniform, iBased, jBased, or CellBased.")


# Takes a legend table and appends it to a 2D array so that it will only appear as part of a single column in the plot array
def offsetLegends(offsetAmt, bareLegend):
    return [[False]*(offsetAmt-1)+[x] for x in bareLegend]

# Determines the correct object by evaluating the subplot architecture to find the actual target axis
def getAxisObject(axes, iLen, jLen, i, j):
    if iLen>1 and jLen>1:
        return axes[i][j]
    elif iLen>1:
        return axes[i]
    elif jLen>1:
        return axes[j]
    else:
        return axes

# Generates and saves a plot array for the given plot data
def generatePlots(xValues, plotData,
    plotXTitles=[], globalXTitle="", localXTitle="",
    plotYTitles=[], globalYTitle="", localYTitle="",
    plotFileName="plotFile", fileType=".png",
    colorHues=[], colorOffset=0,
    specialOpts=[],
    grid=False,
    fixedX=None, fixedY=None,
    spacer=None
):
    iLen = len(plotData)
    jLen = len(plotData[0])

    if not colorHues:
        colorHues = makeHueArray(
            {
                "Type" : "CellBased",
                "GroupData" : [[len(plotData[i][j]) for j in range(jLen)] for i in range(iLen)]
            },
            iLen,
            jLen,
            unique=False,
            offset=colorOffset
        )

    xBreaks = []
    lastValue = 0
    if spacer!=None:
        for i, x in enumerate(xValues[:-1]):
            if xValues[i+1] - x > spacer:
                xBreaks.append((lastValue, i+1))
                lastValue = i+1
    xBreaks.append((lastValue, len(xValues)))

    # Create subplots and add data to subplots
    fig, axes = plt.subplots(iLen, jLen, sharex=True, sharey='row', figsize=(max([3*jLen, 9]),max([2*iLen, 5])))
    for i in range(iLen):
        for j in range(jLen):
            curAxis = getAxisObject(axes, iLen, jLen, i, j)
            curAxis.ticklabel_format(axis='both', scilimits=[-4,4])

            plt.subplot(iLen, jLen, i*jLen+j+1)
            if fixedX!=None:
                plt.xlim(fixedX[0], fixedX[1])
            if fixedY!=None:
                plt.ylim(fixedY[0], fixedY[1])

            for groupInd, group in enumerate(plotData[i][j]):
                shadeNorm = .8 - .4/(1+len(group))
                for shadeInd, itm in enumerate(group):
                    if type(colorHues[i][j][groupInd]) is tuple:
                        colorHue = colorHues[i][j][groupInd][0]
                        shadeVal = colorHues[i][j][groupInd][1]
                    else:
                        colorHue = colorHues[i][j][groupInd]
                        shadeVal = shadeNorm*(.5-((shadeInd+.5)/len(group))**1.5)+.5

                    for xBreak in xBreaks:
                        if xBreak[0] >= len(itm):
                            break
                        pltStartEnd = [xBreak[0], min(xBreak[1], len(itm))]
                        plt.plot(xValues[pltStartEnd[0]:pltStartEnd[1]], itm[pltStartEnd[0]:pltStartEnd[1]], linewidth=2, color=getVector3FromHueAndBrightness(colorHue, shadeVal))
                    if grid:
                        plt.grid()



    # Optional post-processing to add characteristics to subplots
    if specialOpts:
        for req in specialOpts:
            # Adds labels to maximums in data
            if req[0]=="Maximum":
                for i in req[1]:
                    for j in range(jLen):
                        plt.subplot(iLen, jLen, i*jLen+j+1)
                        for x in req[2]:
                            locData = plotData[i][j][x]
                            locMax = max(locData)
                            indMax = [indX for indX, itm in enumerate(locData) if itm==locMax]
                            xPt = xValues[indMax[0]]/3600
                            plt.plot(xPt, locMax, markersize=20, color='black')

                            targAnnotate = getAxisObject(axes, iLen, jLen, i, j)
                            targAnnotate.annotate("("+str(GU.roundToSigFig(xPt, 3))+", "+str(GU.roundToSigFig(locMax,3))+")", xy=(xPt, locMax), textcoords='data')

            elif req[0]=="Legend":
                for i, row in enumerate(req[1]):
                    for j, legendObjs in enumerate(row):
                        if legendObjs:
                            plt.subplot(iLen, jLen, i*jLen+j+1)
                            plt.legend(legendObjs)

    # Add supporting title info along the tops of the graphs
    for i, xTitle in enumerate(plotXTitles):
        if xTitle:
            plt.subplot(iLen, jLen, i+1)
            plt.title(xTitle, fontsize=10)

    # Add supporting title info along the right of the graphs
    for i, yTitle in enumerate(plotYTitles):
        if yTitle:
            curRightAxis = getAxisObject(axes, iLen, jLen, i, -1).twinx()
            curRightAxis.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
            curRightAxis.set_ylabel(yTitle, labelpad=-15, rotation=270)

    # Give supporting information labels along the top, left, right, and bottom of the graphs
    globalAx = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
    plt.xticks([]) # This should be an invisible shell to place captions, no ticks!!
    plt.yticks([])

    if localXTitle:
        plt.xlabel(localXTitle, labelpad=23, fontsize=13)
    if localYTitle:
        plt.ylabel(localYTitle, labelpad=30, fontsize=13)
    if globalXTitle:
        plt.title(globalXTitle, pad=25, fontsize=13)
    if globalYTitle:
        globalRightAxis = globalAx.twinx()
        globalRightAxis.set_frame_on(False)
        globalRightAxis.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
        globalRightAxis.set_ylabel(globalYTitle, rotation=270, fontsize=13, labelpad=15)

    #plt.show()
    #Generates the file instance before writing to it, also prevents accidental overwrite since opening this figure as a new file will cause an error if one exists
    newFig = open(plotFileName+fileType,'x')
    newFig.close()

    fig.savefig(plotFileName+fileType, bbox_inches='tight', dpi=180, transparent=True)
