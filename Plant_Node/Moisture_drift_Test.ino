/*
Moisture drift testing:

so I would personally recommend using just a pure plant pot because then you have an actuall proper percentage rather than seeing the percentage go from 100% to 102%

but before doing this test its probably best to re calibrate it in case the room/area is a lot more humid so first:

1. hold the moisture sensor in the air and read what the measurement is 

2. hold the sensor in water and see what the measurment is

3. update the variables at the top

Using:
So this code should run for 3000 measurments (750 seconds so around 13 minutes) but feel free to change the amount of measurements in the for loop or increase the delay

but the way it will work is that the moment you plug this in you want to have your laptop and the serial monitor running, then when its finished you press the button that looks 
like a page and copy it into a notepad file and save it, i can do the rest from there

Thanks ❤️❤️❤️❤️❤️😍😍😍😍😍🥰🥰🥰🥰🥰🥰
*/

#include <SPI.h>
#include <RH_RF95.h>

int iteration = 0;

const int AirValue   = 605; //calibration of air value and water value
const int WaterValue = 31;

int soilMoistureValue = 0;
int moisturePercent   = 0;

void setup() {
  Serial.begin(9600);
  delay(200);

}

void loop() {

  if(iteration <= 3000){

    soilMoistureValue = analogRead(A0);
    moisturePercent = 100.0 * (AirValue - soilMoistureValue) / (AirValue - WaterValue); //calculate percentage
    moisturePercent = constrain(moisturePercent, 0, 100); //constrain it to 0-100 

    Serial.print(iteration);
    Serial.print(",");
    Serial.println(moisturePercent);

    iteration = iteration + 1;

  }
  else{
    while(1);
  }

  delay(250);
}