#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 21:29:02 2025

@author: danielvalmassei
"""

import serial
import csv
import time

a1 = 100.624
b1 = -0.37679
c1 = -20.5623
d1 = 0.0348656

a2 = 0.1031
b2 = -0.3986
c2 = -0.02322
d2 = 0.07438
e2 = 0.07229
f2 = -0.006866

a3 = -0.02585
b3 = 0.03767
c3 = 0.04563
d3 = 0.1151
e3 = -0.04158
f3 = 0.008737

def convert_voltage(voltage):
    
    
    if voltage >= 4.94:
        numerator = a1 + c1*voltage
        denom = 1 + b1*voltage + d1*voltage**2
        
        return numerator/denom
    
    elif voltage >= 2.842:
        numerator = a2 + c2*voltage + e2*voltage**2
        denom = 1 + b2*voltage + d2*voltage**2 + f2*voltage**3
        return numerator/denom
    
    elif voltage >= 0.375:
        return a3 + b3*voltage + c3*voltage**2 + d3*voltage**3 + e3*voltage**4 + f3*voltage**5
    
    
    else: #something went wrong. return an unreasonable value
        return 1000
    
    
def main():
    # Adjust these parameters
    port = '/dev/tty.usbmodemF0F5BD5329182'  # Replace with your Arduino's port
    baud_rate = 9600
    output_file = 'out.csv'
    
    
    # Initialize serial connection
    ser = serial.Serial(port, baud_rate, timeout=1)
    time.sleep(2)  # Wait for connection
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time (s)','voltage (V)','Pressure (Torr)'])
    
        print('Collecting data... Press ctrl+c to stop.')
        try: 
            while True:
                line = ser.readline().decode('utf-8').strip().split(', ')
                timeNow = line[0]
                voltage = line[1]
                pressure = convert_voltage(float(voltage))
                
                string = [f'{timeNow}',f'{voltage}',f'{pressure}']
                print(string)
                writer.writerow(string)
                
        except KeyboardInterrupt:
            print('Data collection stopped.')
        finally:
            ser.close()

if __name__ == '__main__':
    main()
