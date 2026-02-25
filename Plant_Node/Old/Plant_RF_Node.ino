#include <SPI.h>
#include <RF24.h>
#include <nRF24L01.h>

const int AirValue   = 605;
const int WaterValue = 31;

int soilMoistureValue = 0;
int moisturePercent   = 0;

String ID = "0002";
int Status = 0;

RF24 radio(7, 8);                 // CE, CSN
const byte address[6] = "00001";  // must match receiver
// for this use a different address for each node so then have 00002 for node 2

void setup() {
  Serial.begin(9600);

  radio.begin();
  radio.openWritingPipe(address);
  radio.setPALevel(RF24_PA_MIN);
  radio.stopListening();

  Serial.println("Transmitter ready");
}

void loop() {
  soilMoistureValue = analogRead(A0);

  moisturePercent = 100.0 * (AirValue - soilMoistureValue) / (AirValue - WaterValue);
  moisturePercent = constrain(moisturePercent, 0, 100);

  if (moisturePercent >= 60) {
    Serial.println("Overwatered, possible root rot");
    Status = 3;
  } else if (moisturePercent >= 40) {
    Serial.println("Soil is moist");
    Status = 2;
  } else {
    Serial.println("Watering needed");
    Status = 1;
  }

  // Build message like: 0001,2
  String message = ID + "," + String(Status);

  // Convert to char[] and transmit as a proper C-string
  char payload[32];
  message.toCharArray(payload, sizeof(payload));
  radio.write(payload, strlen(payload) + 1);   // +1 includes null terminator

  delay(1000);
}
