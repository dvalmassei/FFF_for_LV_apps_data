#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:49:25 2024

@author: danielvalmassei
"""

import serial
import csv
import time

filament_type = "PAHT+CF+HeatTreat"
infill_percentage = "90"
layer_height = "0.2mm"
run_num = 1

# Adjust these parameters
port = '/dev/tty.usbmodemF0F5BD5329182'  # Replace with your Arduino's port
baud_rate = 9600
output_file = filament_type + "_" +  infill_percentage + "_" + layer_height + "_" + str(run_num) + '.csv'


# Initialize serial connection
ser = serial.Serial(port, baud_rate, timeout=1)
time.sleep(2)  # Wait for connection

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    print('Collecting data... Press Ctrl+C to stop.')
    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line == 'Opening valve...':
                print(line)
            elif line == 'Press button to start...':
                print(line)
            elif line == 'Closing valve...':
                print(line) 
            elif line == 'Data collection complete.':
                print(line)
                break
            elif line:
                print(line)
                writer.writerow(line.split(', '))
    except KeyboardInterrupt:
        print('Data collection stopped.')
    finally:
        ser.close()
