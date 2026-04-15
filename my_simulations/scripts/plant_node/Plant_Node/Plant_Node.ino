#include <SPI.h>
#include <RH_RF95.h>

//const int AirValue   = 605; //calibration of air value and water value
//const int WaterValue = 31;

const int AirValue   = 593; //calibration of air value and water value sope's values
const int WaterValue = 43;

int soilMoistureValue = 0;
int moisturePercent   = 0;

String ID = "0001"; //ID of plant node - change for each plant node
int Status = 0;

//set pins for cs, rst, G0, and freq
#define RFM95_CS   10
#define RFM95_RST  9
#define RFM95_INT  2

#define RF95_FREQ  433.0

RH_RF95 rf95(RFM95_CS, RFM95_INT);

void resetRadio() { //manually resets radio
  pinMode(RFM95_RST, OUTPUT);
  digitalWrite(RFM95_RST, HIGH);
  delay(10);
  digitalWrite(RFM95_RST, LOW);
  delay(10);
  digitalWrite(RFM95_RST, HIGH);
  delay(10);
}

void setup() {
  Serial.begin(9600);
  delay(200);

  resetRadio(); 

  if (!rf95.init()) {
    Serial.println("LoRa init failed!");
    while (1);
  }
  if (!rf95.setFrequency(RF95_FREQ)) {
    Serial.println("Frequency set failed!");
    while (1);
  }
  rf95.setTxPower(10, false); //sets power to 10 dB so doesnt use so much current
  Serial.println("LoRa 433 MHz TX Ready");
}

void loop() {
  soilMoistureValue = analogRead(A0);

  moisturePercent = 100.0 * (AirValue - soilMoistureValue) / (AirValue - WaterValue); //calculate percentage
  moisturePercent = constrain(moisturePercent, 0, 100); //constrain it to 0-100 
  Serial.println("Plant moisture: " + String(moisturePercent) + "%");

  if (moisturePercent >= 60) { //set moisture thresholds
    Serial.println("Overwatered, possible root rot");
    Status = 3;
  } 
  else if (moisturePercent >= 40) {
    Serial.println("Soil is moist");
    Status = 2;
  } 
  else {
    Serial.println("Watering needed");
    Status = 1;
  }

  // build message 
  String message = ID + "," + String(Status); //create message of ID and status

  char payload[32];
  message.toCharArray(payload, sizeof(payload));//converts string to char array

  rf95.send((uint8_t*)payload, sizeof(payload)); //send radio message to pc receiver Lora
  rf95.waitPacketSent();

  Serial.println("Sent: " + String(payload));
  delay(1000);
}
