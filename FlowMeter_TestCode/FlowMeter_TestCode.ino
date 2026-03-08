volatile double waterFlow;

void setup() {
  Serial.begin(9600);
  waterFlow = 0;
  attachInterrupt(0, pulse, RISING);  //Digital Pin 2: interrupt 0
}

void loop() {
  Serial.print("waterFlow: ");
  Serial.print(waterFlow);
  Serial.println("  L");
  delay(0);
}

void pulse() {
  waterFlow += 1.0 / 450.0; //450 pulses for 1 liter
}
