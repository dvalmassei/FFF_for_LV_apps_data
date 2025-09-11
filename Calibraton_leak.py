#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:28:40 2024

@author: danielvalmassei
"""

import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import convert_voltage as cv
from plot_data import dfToPlot
from scipy.optimize import curve_fit
import numpy as np


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
    
    
    plt.scatter(df0['Time (s)'][moveValveIndex:],pressure0[moveValveIndex:]*1000, s=1, label=runName0, color='Red')
    #plt.scatter(df0['Time (s)'][moveValveIndex+1:] + df0['Time (s)'][moveValveIndex],pressure0[moveValveIndex+1:]*1000, s=1, color='Red')

    plt.ylabel('mTorr')
    plt.xlabel('Time [s]')
    plt.grid(which='both')
    plt.legend()
    plt.show()
    
    x = np.array(df0['Time (s)'][moveValveIndex+1:])
    y = pressure0[moveValveIndex+1:]*1000
    p0 = pressure0[moveValveIndex+1]*1000
    
    print(p0)
    
    def func(x,b,c):
        return (p0-c)*np.exp(-x/b) + c
    
    popt,pcov = curve_fit(func, x[:], y[:], p0=(335,210))
    print(popt, popt[0]/popt[1])
    
    plt.scatter(x,y, s=1,label='fit')
    plt.plot(x,func(x,*popt), color = 'Red',label=runName0)
    plt.grid(which='both')
    plt.ylabel('mTorr')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()
    
    
    plt.plot((y[10:] - y[:-10])*0.5 / (x[10:] - x[:-10]))
    plt.show()
   
    print(np.mean( ((y[10:] - y[:-10])*0.5 / (x[10:] - x[:-10]))[10:] ))
    
if __name__=='__main__':
    main()