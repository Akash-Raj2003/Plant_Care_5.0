const unsigned long MESSAGE_DELAY_MS = 3000;

struct PlantAlert {
  int plant_id;
  float moisture;
  int low_moisture;
};

PlantAlert alerts[] = {
  {1, 18.5, 1},
  {3, 21.0, 1},
  {1, 17.9, 1},
  {4, 35.0, 0},
  {2, 19.2, 1},
  {3, 20.8, 1},
  {4, 18.1, 1}
};

const int ALERT_COUNT = sizeof(alerts) / sizeof(alerts[0]);
int current_alert = 0;

void printAlert(const PlantAlert& alert) {
  Serial.print(alert.plant_id);
  Serial.print(",");
  Serial.print(alert.moisture, 1);
  Serial.print(",");
  Serial.println(alert.low_moisture);
}

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ;  // Wait for serial on boards that require it.
  }

  Serial.println("# Test_MoistureQueue started");
  Serial.println("# Format: plant_id,moisture,low_moisture");
}

void loop() {
  printAlert(alerts[current_alert]);
  current_alert = (current_alert + 1) % ALERT_COUNT;
  delay(MESSAGE_DELAY_MS);
}
