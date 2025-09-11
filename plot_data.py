#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:31:40 2024

@author: danielvalmassei
"""

import convert_voltage as cv
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


def dfToPlot(df):
    pressure0 = np.zeros(len(df['Time (s)']))
    moveValveIndex = []


    for i in range(len(df['Time (s)'])):
        pressure0[i] = cv.convert_voltage(df['Voltage (V)'][i])
        if df['Time (s)'][i] >= 600.0: #find the index of the time when the valve was moved
            moveValveIndex.append(i)

    moveValveIndex = moveValveIndex[0]
    
    return pressure0, moveValveIndex
    
    
def main():
    plt.style.use([hep.style.LHCb2]) #let's make our plots pretty
    
    runName0 = 'Calibration_Cap_YorLok_1'

    
    df0 = pd.read_csv(runName0 + '.csv')


        
    pressure0 = np.zeros(len(df0['Time (s)']))


    moveValveIndex = []
    for i in range(len(df0['Time (s)'])):
        pressure0[i] = cv.convert_voltage(df0['Voltage (V)'][i])

        if df0['Time (s)'][i] >= 600.0:
            moveValveIndex.append(i)

    moveValveIndex = moveValveIndex[0]
    
    plt.scatter(df0['Time (s)'][:moveValveIndex],pressure0[:moveValveIndex]*1000, s=1, label=runName0, color='Red')
    plt.scatter(df0['Time (s)'][moveValveIndex+1:] + df0['Time (s)'][moveValveIndex],pressure0[moveValveIndex+1:]*1000, s=1, color='Red')

    plt.yscale('log')
    plt.ylabel('mTorr')
    plt.xlabel('Time [s]')
    plt.ylim((10,2000))
    plt.grid(which='both')
    plt.legend()
    plt.show()
    
    
if __name__ =='__main__':
    main()