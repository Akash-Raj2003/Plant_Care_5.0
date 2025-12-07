
import serial

ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1) #open serial port for arduino
print("Opened /dev/ttyACM0, waiting for lines...")

try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        #reads whatever arduino prints to serial
        if line:
            print("Got:", line)
except KeyboardInterrupt:
    print("Exiting")
finally:
    ser.close()#closes serial when ctrl + c pressed


