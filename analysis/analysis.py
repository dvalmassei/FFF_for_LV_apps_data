#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:31:40 2024

@author: danielvalmassei
"""

import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from plot_data import dfToPlot
from scipy.optimize import curve_fit
import numpy as np

def func(x,a,b,c):
    return a * np.exp(-x/b) + c

def func2(x,p0,b):
    return p0 * (1 - np.exp(-x/b))


def main():
    plt.style.use([hep.style.LHCb2]) #let's make our plots pretty
    
    calibrationRun = 'Calibration_Cap_YorLok_1'
    
    runNames = [
                'PAHT+CF_50_0.2mm_0','PAHT+CF_50_0.2mm_1','PAHT+CF_50_0.2mm_2',
                'PAHT+CF_75_0.2mm_0','PAHT+CF_75_0.2mm_1','PAHT+CF_75_0.2mm_2',
                'PAHT+CF_90_0.2mm_0',
                'PC_50_0.2mm_0','PC_50_0.2mm_1','PC_50_0.2mm_2',
                'PC_75_0.2mm_0','PC_75_0.2mm_1','PC_75_0.2mm_2',
                'PC_90_0.2mm_0','PC_90_0.2mm_1','PC_90_0.2mm_2',
                'PPS+CF_50_0.2mm_0','PPS+CF_50_0.2mm_1','PPS+CF_50_0.2mm_2',
                'PPS+CF_75_0.2mm_0','PPS+CF_75_0.2mm_1','PPS+CF_75_0.2mm_2',
                'PPS+CF_90_0.2mm_0','PPS+CF_90_0.2mm_1','PPS+CF_90_0.2mm_2',
                'PC+ABS_50_0.2mm_0','PC+ABS_50_0.2mm_1','PC+ABS_50_0.2mm_2',
                'PC+ABS_75_0.2mm_0','PC+ABS_75_0.2mm_1','PC+ABS_75_0.2mm_2',
                'PC+ABS_90_0.2mm_0','PC+ABS_90_0.2mm_1','PC+ABS_90_0.2mm_2',
                'ABS+CF_50_0.2mm_0','ABS+CF_50_0.2mm_1','ABS+CF_50_0.2mm_2',
                'ABS+CF_75_0.2mm_0','ABS+CF_75_0.2mm_1','ABS+CF_75_0.2mm_2',
                'ABS+CF_90_0.2mm_0','ABS+CF_90_0.2mm_1','ABS+CF_90_0.2mm_2',
                'ABS+CF+AcetoneBath_50_0.2mm_0','ABS+CF+AcetoneBath_50_0.2mm_1','ABS+CF+AcetoneBath_50_0.2mm_2',
                'ABS+CF+AcetoneBath_75_0.2mm_0','ABS+CF+AcetoneBath_75_0.2mm_1','ABS+CF+AcetoneBath_75_0.2mm_2',
                'ABS+CF+AcetoneBath_90_0.2mm_0','ABS+CF+AcetoneBath_90_0.2mm_1','ABS+CF+AcetoneBath_90_0.2mm_2',
                'PC+ABS+AcetoneBath_50_0.2mm_0','PC+ABS+AcetoneBath_50_0.2mm_1','PC+ABS+AcetoneBath_50_0.2mm_2','PC+ABS+AcetoneBath_50_0.2mm_3',
                'PC+ABS+AcetoneBath_75_0.2mm_0','PC+ABS+AcetoneBath_75_0.2mm_1','PC+ABS+AcetoneBath_75_0.2mm_2',
                'PC+ABS+AcetoneBath_90_0.2mm_0','PC+ABS+AcetoneBath_90_0.2mm_1','PC+ABS+AcetoneBath_90_0.2mm_2',
                'PAHT+CF+HeatTreat_50_0.2mm_0','PAHT+CF+HeatTreat_50_0.2mm_1','PAHT+CF+HeatTreat_50_0.2mm_2',
                'PAHT+CF+HeatTreat_75_0.2mm_0','PAHT+CF+HeatTreat_75_0.2mm_1','PAHT+CF+HeatTreat_75_0.2mm_2',
                'PAHT+CF+HeatTreat_90_0.2mm_0','PAHT+CF+HeatTreat_90_0.2mm_1','PAHT+CF+HeatTreat_90_0.2mm_2',
                'PC+MCBath_50_0.2mm_0','PC+MCBath_50_0.2mm_1','PC+MCBath_50_0.2mm_2',
                'PC+MCBath_75_0.2mm_0','PC+MCBath_75_0.2mm_1','PC+MCBath_75_0.2mm_2',
                'PC+MCBath_90_0.2mm_0','PC+MCBath_90_0.2mm_1','PC+MCBath_90_0.2mm_2'
                ]
    
    colors = plt.cm.rainbow(np.linspace(0,1,2*len(runNames)+1)) #set up an array of colors for our plots
    
    # Initialize arrays for storage of our fit variables
    a = np.zeros(len(runNames)+1)
    b = np.zeros(len(runNames)+1)
    c = np.zeros(len(runNames)+1)
    
    a1 = np.zeros(len(runNames))
    b1 = np.zeros(len(runNames))
    c1 = np.zeros(len(runNames))
    
    # read in the calibration data and convert the votages to pressures
    calibrationDf = pd.read_csv(calibrationRun + '.csv')
    calibrationPressures, calibrationMoveValveIndex = dfToPlot(calibrationDf)
    p0 = calibrationPressures[calibrationMoveValveIndex+1]*1000
    
    popts,pcovs = curve_fit(func, calibrationDf['Time (s)'][1000:calibrationMoveValveIndex],
                            calibrationPressures[1000:calibrationMoveValveIndex]*1000, 
                            p0=(p0,100,calibrationPressures[calibrationMoveValveIndex]))
    a[0],b[0],c[0] = popts[0], popts[1], popts[2]
    print(popts)
    
    plt.figure(figsize=(40, 24))

    plt.scatter(calibrationDf['Time (s)'][:calibrationMoveValveIndex]/60,
                calibrationPressures[:calibrationMoveValveIndex]*1000, 
                s=1, label=f"Calibration, b = {b[0]:.4}, c = {c[0]:.4}",
                color = colors[0])
    #plt.plot(calibrationDf['Time (s)'][:calibrationMoveValveIndex]/60, 
    #         func(calibrationDf['Time (s)'][:calibrationMoveValveIndex],*popts),
    #         color = colors[2])
    
    
    # Read the data files, convert voltages to pressures, fit the data, then plot everything
    for i in range(len(runNames)):
        df = pd.read_csv(runNames[i] + '.csv')
        
        pressures, moveValveIndex = dfToPlot(df)
        p0 = pressures[0]
        
        # Subtract away the calibration
        #pressures = pressures - calibrationPressures
        
        # Fit the data to our functions defined above
        
        popts,pcovs = curve_fit(func, df['Time (s)'][1000:moveValveIndex],
                                pressures[1000:moveValveIndex]*1000, 
                                p0=(1000,100,100))
        a[i+1],b[i+1],c[i+1] = popts[0],popts[1],popts[2]
        
        label = f"{runNames[i]}, b = {b[i+1]:.4}, c = {c[i+1]:.4}"
        
        # Plot the current data
        plt.scatter(df['Time (s)'][:moveValveIndex]/60,pressures[:moveValveIndex]*1000, s=1, label=label, color = colors[2*i+1])
        #plt.plot(df['Time (s)'][:moveValveIndex]/60, func(df['Time (s)'][:moveValveIndex],*popts), color = colors[2*i+2])
        #plt.plot(df['Time (s)'][:moveValveIndex]/60, log(df['Time (s)'][:moveValveIndex],a1[i],b1[i],c1[i]), color = colors[3*i+2])
        plt.scatter((df['Time (s)'][moveValveIndex+1:] + df['Time (s)'][moveValveIndex])/60,pressures[moveValveIndex+1:]*1000, s=1, color = colors[2*i+1])
    
    # Finish making the plot
    plt.yscale('log')
    plt.ylabel('mTorr')
    plt.xlabel('Time [min]')
    #plt.ylim((20,800))
    plt.ylim((20,30000))
    plt.grid(which='both')
    plt.legend(frameon = True, fontsize='large', framealpha=0.95, bbox_to_anchor=(1.01,1))
    plt.show()
    
    
    #print((c - c[0])*0.5 /750.06167)
    
    '''
    plt.bar(np.linspace(0,len(runNames)+1,len(runNames)+1),(c - c[0])*0.105 /750.06167)
    plt.yscale('log')
    plt.ylabel('mbar * l / s')
    plt.title('Leak Rate')
    plt.xlabel('runName index')
    plt.show()
    '''
    
    S = 0.713 / b[0] #0.00357356 #l * s^-1
    Q1 = c[0] * S #0.0797619 #mTorr * l * s^-1
    print(S, Q1/750.06167)
    leakRate = ((c[1:]*S) - Q1) / 750.06167 #((c[1:]*S*Q1) /(c[:1]*S - Q1)) /750.06167 #(c[1:] - c[0])*0.105 /750.06167
    
    #Sort the runs
    splitRunNames = []
    for i in range(len(runNames)):
        splitRunNames.append(runNames[i].split('_'))
        splitRunNames[i].append(leakRate[i])
        splitRunNames[i].append(a[i+1])
        splitRunNames[i].append(b[i+1])
        splitRunNames[i].append(c[i+1])



            
    df = pd.DataFrame(splitRunNames, columns = ['Filament','Infill Percentage', 
                                                'Layer Height','Sample No.','Leak Rate','A','B','C'] )
    
    df['Infill Percentage'] = df['Infill Percentage'].astype(int)
    
    
    infills = df['Infill Percentage'].unique()
    filaments = ['ABS+CF','ABS+CF+AcetoneBath','PAHT+CF','PAHT+CF+HeatTreat','PC','PC+MCBath','PC+ABS',
                 'PC+ABS+AcetoneBath','PPS+CF']#np.sort(df['Filament'].unique())
    
    colors = plt.cm.gist_rainbow(np.linspace(0,1,len(infills)*len(filaments) + 1)) #set up an array of colors for our plots


    x = np.arange(len(infills))  # the label locations
    width = 1/(len(filaments)+1) # the width of the bars

    plt.figure(figsize=(15, 12))
    
    handles = []
    for i in range(len(infills)):
        for j in range(len(filaments)):
            entries = df.loc[(df['Infill Percentage']==infills[i]) & (df['Filament']==filaments[j]), 'Leak Rate']
            print(filaments[j], infills[i], np.mean(entries), np.std(entries))
            
            bar = plt.bar(x[i] + j*width, np.mean(entries), width, 
                           color = colors[len(infills)*j],
                           yerr = np.std(entries), ecolor ='black', capsize = 10)
            handles.append(bar,)
    #plt.figure().set_figwidth(15)    
    plt.yscale('log')
    plt.title('Leak Rate')
    plt.ylabel(r'$mbar \cdot l \cdot s^{-1}$')
    plt.xlabel('Infill Overlap')
    plt.xticks(x + width*(len(filaments)-1)/2, infills)
    plt.legend(handles[:len(filaments)],filaments, frameon=True, bbox_to_anchor=(0.9,0.8), fontsize='medium')
    plt.ylim((1e-5,0.025))
    plt.show()
    
    df.to_csv('out.csv', index=False)
          
    
    

if __name__ == '__main__':
    main()
    
    