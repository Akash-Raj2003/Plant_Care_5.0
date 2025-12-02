Pump code advanced

#include "GravityPump.h"
#include "Button.h" // https://github.com/DFRobot/Button

GravityPump pump;
Button button;
bool run = true;
int debug = 1;

void setup()
{
  pump.setPin(9);
  button.init(2);
  Serial.begin(115200);
  pump.getFlowRateAndSpeed();
}

void loop()
{
  pump.update();
  button.update();
  if (debug)
  {
    // in debug mode the pump will do calibration.
    // if set the debug mode off then the function works.
    pump.calFlowRate();
  }
  else
  {
    if (run)
    {
      // switch the function by using Comments.
      run = false;
      // Serial.println(pump.flowPump(6.6));
      // Just put the number in ml then the pump will dosing the numbers of liquid
      // and you can find the numbers from serial port.
      Serial.println(pump.timerPump(120000));
      // Just put the number in millisecond then the pump will dosing the time of
      // and you can find the numbers from serial port.
    }
  }

  if (button.click())
  {
    // Serial.println("click");
    // when you click the button the pump will stop immediately
    pump.stop();
  }

  if (button.press())
  {
    // (rest of code was cut off in the image)
  }
}

