#!/usr/bin/env python3
import serial,time,signal,sys,argparse
from datetime import datetime


continu=True
def signal_handler(sig, frame):
    print('Ctrl+C detected. Exiting...')
    global continu
    continu = False

signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser(description='pm reader.')
parser.add_argument('-p', '--port',type=str,default='/dev/ttyUSB1')
args = parser.parse_args()

ser = serial.Serial(args.port,timeout=1)
buf = []
outfile = open('co2.csv', "a+") 

while continu:
    ser.write(b'\xFF')
    ser.write(b'\x01')
    ser.write(b'\x86')
    ser.write(b'\x00')
    ser.write(b'\x00')
    ser.write(b'\x00')
    ser.write(b'\x00')
    ser.write(b'\x00')
    ser.write(b'\x79') # checksum
    buf = ser.read(9)

    if len(buf) >= 9:
        co2 = buf[2]*256 + buf[3]
        now = datetime.now()
        print(now.strftime('%m/%d/%Y %H:%M:%S, ') + 'co2: ' + str(co2) + ' ppm')
        outfile.write(now.strftime('%m/%d/%Y %H:%M:%S;') + str(co2) + '\n')
        outfile.flush()
        time.sleep(5)
    else:
        print("Time ou!    " + str(buf))


outfile.close()