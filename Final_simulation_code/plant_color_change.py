from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

client = RemoteAPIClient()
sim = client.getObject('sim')

# plant model in the scene has 14 instances named indoorPlant[0] to indoorPlant[13]. Each plant is a compound shape with a LEAVES color group that we want to change. The plant_map dictionary maps user-friendly plant names (plant1, plant2, ..., plant14) to the corresponding object paths in the CoppeliaSim scene. This allows us to easily reference each plant by name when we want to toggle its color.
plant_map = {
    "plant1": "/indoorPlant[0]/visible",
    "plant2": "/indoorPlant[1]/visible",
    "plant3": "/indoorPlant[2]/visible",
    "plant4": "/indoorPlant[3]/visible",
    "plant5": "/indoorPlant[4]/visible",
    "plant6": "/indoorPlant[5]/visible",
    "plant7": "/indoorPlant[6]/visible",
    "plant8": "/indoorPlant[7]/visible",
    "plant9": "/indoorPlant[8]/visible",
    "plant10": "/indoorPlant[9]/visible",
    "plant11": "/indoorPlant[10]/visible",
    "plant12": "/indoorPlant[11]/visible",
    "plant13": "/indoorPlant[12]/visible",
    "plant14": "/indoorPlant[13]/visible",
}

# The LEAF_COLOR_NAME is the name of the color group in the compound shape that we want 
# to change. In CoppeliaSim, compound shapes can have multiple color
#  groups (e.g., for different parts of the model), and we specify which one to 
# change by name when calling sim.setShapeColor.
LEAF_COLOR_NAME = "LEAVES"

# Yellow color 
YELLOW = [1.00, 1.00, 0.59]

# A green close to the original leaves
GREEN = [0.598, 1.00, 0.59]

# Toggle state
plant_is_yellow = {name: False for name in plant_map}


def set_plant_leaves_color(plant_name, rgb):
    if plant_name not in plant_map:
        print(f"{plant_name} not found")
        return False

    try:
        shape_handle = sim.getObject(plant_map[plant_name])

        # Change only the LEAVES color group of the compound shape
        sim.setShapeColor(
            shape_handle,
            LEAF_COLOR_NAME,
            sim.colorcomponent_ambient_diffuse,
            rgb
        )
        return True

    except Exception as e:
        print(f"Could not change {plant_name}: {e}")
        return False


def toggle_plant(plant_name):
    if plant_name not in plant_map:
        print("Invalid plant name")
        return

    if plant_is_yellow[plant_name]:
        ok = set_plant_leaves_color(plant_name, GREEN)
        if ok:
            plant_is_yellow[plant_name] = False
            print(f"{plant_name} restored to green")
    else:
        ok = set_plant_leaves_color(plant_name, YELLOW)
        if ok:
            plant_is_yellow[plant_name] = True
            print(f"{plant_name} changed to yellow")


sim.startSimulation()
time.sleep(0.5)

try:
    print("Type plant1 ... plant14 to toggle that plant's leaves")
    print("Type q to quit")

    while True:
        cmd = input("Enter command: ").strip().lower()

        if cmd == "q":
            break

        if cmd in plant_map:
            toggle_plant(cmd)
        else:
            print("Invalid command")

finally:
    sim.stopSimulation()