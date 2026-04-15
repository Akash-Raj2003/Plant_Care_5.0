#!/usr/bin/env python3

import json
import serial
import rospy
from serial.serialutil import SerialException
from plant_msgs.msg import PlantMoisture
from std_msgs.msg import Float32
import paho.mqtt.client as mqtt

RELATIVE_HUMIDITY_TOPIC = "/plant/relative_humidity"
N_DILUTION_RATIO_TOPIC = "/plant/n_dilution_ratio"
N_CONCENTRATION_TOPIC = "/plant/n_concentration"


def relative_humidity_callback(msg):
    relative_humidity_callback.current_value = float(msg.data)


relative_humidity_callback.current_value = None


def n_dilution_ratio_callback(msg):
    n_dilution_ratio_callback.current_value = float(msg.data)


n_dilution_ratio_callback.current_value = None


def n_concentration_callback(msg):
    n_concentration_callback.current_value = float(msg.data)


n_concentration_callback.current_value = None


def parse_plant_line(line):
    parts = [part.strip() for part in line.split(",")]

    if len(parts) == 3:
        plant_id, moisture, low = parts
        low_moisture = bool(int(low))
    elif len(parts) == 2:
        plant_id, moisture = parts
        low_moisture = float(moisture) <= 40.0
    else:
        raise ValueError(
            "expected 'plant_id,moisture,low' or 'plant_id,moisture', got %d fields" % len(parts)
        )

    return int(plant_id), float(moisture), low_moisture


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        rospy.loginfo("Connected to MQTT broker successfully.")
    else:
        rospy.logwarn("Failed to connect to MQTT broker. Return code: %d", rc)


def publish_mqtt_if_changed(mqtt_client, mqtt_topic, mqtt_payload, last_mqtt_payload):
    if mqtt_payload == last_mqtt_payload:
        return last_mqtt_payload

    result = mqtt_client.publish(mqtt_topic, json.dumps(mqtt_payload))

    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        rospy.loginfo("Published to MQTT topic %s: %s", mqtt_topic, mqtt_payload)
        return dict(mqtt_payload)

    rospy.logwarn("Failed to publish to MQTT topic %s", mqtt_topic)
    return last_mqtt_payload


def main():
    rospy.init_node('plant_radio_bridge_node')
    rospy.loginfo("Plant radio bridge node started.")

    # -------------------------
    # Serial settings
    # -------------------------
    serial_port = '/dev/ttyUSB0'
    serial_baud = 9600

    # -------------------------
    # MQTT settings
    # -------------------------
    mqtt_broker = "192.168.241.85"      # Change this to your broker IP if needed
    mqtt_port = 1883
    # Open serial connection to Arduino
    try:
        ser = serial.Serial(serial_port, serial_baud, timeout=1)
        rospy.loginfo("Opened serial port %s", serial_port)
    except SerialException as e:
        rospy.logerr("Failed to open serial port %s: %s", serial_port, e)
        return

    # Create ROS publisher
    pub = rospy.Publisher("/plant/moisture_alert", PlantMoisture, queue_size=10)
    rospy.Subscriber(
        RELATIVE_HUMIDITY_TOPIC,
        Float32,
        relative_humidity_callback,
    )
    rospy.Subscriber(
        N_DILUTION_RATIO_TOPIC,
        Float32,
        n_dilution_ratio_callback,
    )
    rospy.Subscriber(
        N_CONCENTRATION_TOPIC,
        Float32,
        n_concentration_callback,
    )

    # Set up MQTT client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect

    try:
        mqtt_client.connect(mqtt_broker, mqtt_port, 60)
        mqtt_client.loop_start()  # Start MQTT background networking loop
    except Exception as e:
        rospy.logerr("Failed to connect to MQTT broker at %s:%d - %s", mqtt_broker, mqtt_port, e)
        return

    last_logged_publish = None
    last_mqtt_payload_by_topic = {}

    while not rospy.is_shutdown():
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                rospy.loginfo("Received: %s", line)

                plant_id, moisture, low = parse_plant_line(line)
                rospy.loginfo(
                    "Logged - Plant ID: %s, Moisture: %s, Low: %s",
                    plant_id,
                    moisture,
                    low
                )

                # Create ROS message
                msg = PlantMoisture()
                msg.plant_id = plant_id
                msg.moisture_level = moisture
                msg.low_moisture = low

                # Publish to ROS topic
                pub.publish(msg)

                # Create MQTT payload
                mqtt_payload = {
                    "plant_id": msg.plant_id,
                    "moisture_level": msg.moisture_level,
                    "low_moisture": msg.low_moisture,
                    "relative_humidity": relative_humidity_callback.current_value,
                    "n_dilution_ratio": n_dilution_ratio_callback.current_value,
                    "n_concentration": n_concentration_callback.current_value,
                }
                mqtt_topic = f"/plant/moisture_alert_{msg.plant_id}"

                last_mqtt_payload_by_topic[mqtt_topic] = publish_mqtt_if_changed(
                    mqtt_client,
                    mqtt_topic,
                    mqtt_payload,
                    last_mqtt_payload_by_topic.get(mqtt_topic),
                )

                publish_key = (
                    msg.plant_id,
                    msg.moisture_level,
                    msg.low_moisture,
                    relative_humidity_callback.current_value,
                    n_dilution_ratio_callback.current_value,
                    n_concentration_callback.current_value,
                )
                if publish_key != last_logged_publish:
                    rospy.loginfo(
                        "Published on /plant/moisture_alert once: plant_id=%d moisture_level=%.1f low_moisture=%s relative_humidity=%s n_dilution_ratio=%s n_concentration=%s",
                        msg.plant_id,
                        msg.moisture_level,
                        msg.low_moisture,
                        relative_humidity_callback.current_value,
                        n_dilution_ratio_callback.current_value,
                        n_concentration_callback.current_value,
                    )
                    last_logged_publish = publish_key

            except Exception as e:
                rospy.logwarn("Failed to process line '%s': %s", line, str(e))

    # Cleanup
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    ser.close()


if __name__ == '__main__':
    main()
