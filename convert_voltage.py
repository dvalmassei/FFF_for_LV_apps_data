#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:48:20 2024

@author: danielvalmassei
"""
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

def convert_voltage(voltage):
    if voltage >= 4.94:
        a = 100.624
        b = -0.37679
        c = -20.5623
        d = 0.0348656
        
        numerator = a + c*voltage
        denom = 1 + b*voltage + d*voltage**2
        
        return numerator/denom
    
    elif voltage >= 2.842:
        a = 0.1031
        b = -0.3986
        c = -0.02322
        d = 0.07438
        e = 0.07229
        f = -0.006866
        
        numerator = a + c*voltage + e*voltage**2
        denom = 1 + b*voltage + d*voltage**2 + f*voltage**3
        return numerator/denom
    
    elif voltage >= 0.375:
        a = -0.02585
        b = 0.03767
        c = 0.04563
        d = 0.1151
        e = -0.04158
        f = 0.008737
        return a + b*voltage + c*voltage**2 + d*voltage**3 + e*voltage**4 + f*voltage**5
    
    
    else: #something went wrong. return an unreasonable value
        return 1000
        
        
def main():
    plt.style.use(hep.style.LHCb2)

    voltage = np.linspace(0.375, 5.659, num = 1000)
    pressure = np.zeros(len(voltage))
    
    for i in range(len(voltage)):
        pressure[i] = convert_voltage(voltage[i])
        
    plt.plot(voltage, pressure, color= 'Red')
    plt.yscale('log')
    plt.ylabel('Pressure [Torr]')
    plt.xlabel('Voltage [V]')
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':
    main()