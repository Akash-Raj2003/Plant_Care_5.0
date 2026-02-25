#include <SPI.h>
#include <RH_RF95.h>
#include <RF24.h>
#include <nRF24L01.h>

RF24 radio(7, 8);                 // CE, CSN
const byte address[6] = "00001";  // adress of NRF radios, must match transmitter

// Lora pin definitions for cd, rst, and interrupt pin
#define RFM95_CS   10   
#define RFM95_RST  9
#define RFM95_INT  2    

#define RF95_FREQ  433.0 //set frequency to 433 MHz

RH_RF95 rf95(RFM95_CS, RFM95_INT);

void resetRadio() { //force a hard reset of sensor at start of code
  pinMode(RFM95_RST, OUTPUT);
  digitalWrite(RFM95_RST, HIGH); delay(10);
  digitalWrite(RFM95_RST, LOW);  delay(10);
  digitalWrite(RFM95_RST, HIGH); delay(10);
}

void setup() {
  Serial.begin(9600);
  delay(200);

// NRF setup
/////////////////////////////////////////////////////////////////
  if (radio.begin()) {
    Serial.println("NRF Online");
  }
  radio.setChannel(76);
  radio.setDataRate(RF24_1MBPS);
  radio.setPALevel(RF24_PA_LOW);   // start LOW or MIN for testing
  radio.openWritingPipe(address);
  radio.setPALevel(RF24_PA_MIN);
  radio.stopListening();

// Lora setup
/////////////////////////////////////////////////////////////////
  resetRadio();

  if (rf95.init()) { //initialise Loras
    Serial.println("LoRa online");
  }

  if (!rf95.setFrequency(RF95_FREQ)) {
    Serial.println("Frequency set failed!");
    while (1);
  }
  rf95.setTxPower(10, false); //set power to 10dB so it doesnt use too much current

  Serial.println("Dual radio receiver ready");
}

void loop() {
  if (rf95.available()) { //if Loras are available, receive radio packet and send value to python GUI serial
    char payload[32] = {0};
    uint8_t len = sizeof(payload);

    if (rf95.recv((uint8_t*)payload, &len)) {
      Serial.println(payload);  // e.g. 0001,2
    }
  }
   
  if (Serial.available()) { //runs if a serial message is being sent from GUI to arduino
    String message = Serial.readStringUntil('\n');  // reads full message
    message.trim();  // remove whitespace

    if (message == "Stop") { //if emergency stop in GUI was pressed, sends an NRF message to ridgeback
      const char payload[] = "stopping";
      bool ok = radio.write(payload, sizeof(payload)); // includes null terminator

      Serial.println(ok ? "Sent emergency stop to Ridgeback" : "Send failed");
    }

    else if (message.startsWith("RIDGEBACK,")) { //if user sends plant request will send value to ridgeback
    int comma = message.indexOf(',');
    int plantId = message.substring(comma + 1).toInt();  // gets the number after comma

    // Send plant id to ridgeback via radio
    char payload[32];
    snprintf(payload, sizeof(payload), "PLANT,%d", plantId);
    bool ok = radio.write(payload, strlen(payload) + 1);
    Serial.println(ok ? "Sent plant job to Ridgeback" : "Send failed");
    }
  
  }
}
  

