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
parser.add_argument('-p', '--port',type=str,default='/dev/ttyUSB0')
args = parser.parse_args()

ser = serial.Serial(args.port,timeout=1)
buf = []
outfile = open('pm.csv', "a+") 
while continu:
    b = ser.read()
    if (len(b) <= 0):
        print('Timeout!')
    else:
        buf.append(b)
        if b == b'\xab' and len(buf) >= 10:
            if buf[-10] == b'\xaa' and buf[-9] == b'\xc0':
                pm2p5 = (int.from_bytes(buf[-8],"little") + int.from_bytes(buf[-7],"little")*256) / 10.0
                pm10 = (int.from_bytes(buf[-6],"little") + int.from_bytes(buf[-5],"little")*256) / 10.0
                now = datetime.now()
                print(now.strftime('%m/%d/%Y %H:%M:%S, ') + 'pm2.5: ' + str(pm2p5) + ", pm10: " + str(pm10) + ' [ug/m3]')
                outfile.write(now.strftime('%m/%d/%Y %H:%M:%S;') + str(pm2p5) + ";"+str(pm10)+ '\n')
                outfile.flush()

            buf = []
        
    if (len(buf) > 100):
        buf = []

outfile.close()
