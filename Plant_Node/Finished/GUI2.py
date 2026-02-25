#cd "OneDrive - Loughborough University\Year 4\Group Project\Programming"

import tkinter as tk
from PIL import Image, ImageTk
from tkinter import scrolledtext
import serial



#################################################
# Create the main window
##################################################
root = tk.Tk()
root.geometry("1600x1000")  # Set window size
root.title("Greenhouse GUI")  # Set window title


#################################################
# Stop Button
##################################################
def Stop():
    log_event(f"User stopped operation") #update action log by calling log event
    print("Operation stopped")
    if ser:
        ser.write(b"Stop\n") #send "stop" to arduino
        ser.flush()
    else:
        print("No serial connection")

# Creating a stop button that calls Stop when pressed
button = tk.Button(root, 
                   text="Emergency Stop", 
                   command=Stop,
                   activebackground="red",
                   anchor="center",
                   bd=3,
                   bg="lightgray",
                   cursor="hand2",
                   disabledforeground="red",
                   fg="black",
                   font=("Arial", 14, "bold"),
                   height=4,
                   highlightbackground="black",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised",
                   padx=10,
                   pady=15,
                   width=17,
                   wraplength=250)

button.place(x=25, y=0)

#################################################
# Battery Label - Placeholder
#################################################
text_var = tk.StringVar()
text_var.set("Battery: 50%")

# Create the label widget with all options
Battery_lb = tk.Label(root, 
                 textvariable=text_var, 
                 anchor=tk.CENTER,       
                 bg="yellow",      
                 height=4,              
                 width=17,              
                 bd=3,                  
                 font=("Arial", 14, "bold"),   
                 fg="black",             
                 padx=15,               
                 pady=15,                
                 justify=tk.CENTER,    
                 relief=tk.RAISED,           
                 wraplength=250         
                )
# Pack the label into the window
Battery_lb.place(x=300, y=0)

#################################################
# Fill Level Label - Placeholder
#################################################
fill_var = tk.StringVar()
fill_var.set("Tank: Full")

# Create the label widget with all options
Fill_Level_lb = tk.Label(root, 
                 textvariable=fill_var, 
                 anchor=tk.CENTER,       
                 bg="lightblue",      
                 height=4,              
                 width=17,              
                 bd=3,                  
                 font=("Arial", 14, "bold"),   
                 fg="black",             
                 padx=15,               
                 pady=15,                
                 justify=tk.CENTER,    
                 relief=tk.RAISED,           
                 wraplength=250         
                )
# Pack the label into the window
Fill_Level_lb.place(x=550, y=0)

#################################################
# Status Label
#################################################
status_var = tk.StringVar()
status_var.set("Status: Idle")

# Create the label widget with all options
Status_lb = tk.Label(root, 
                 textvariable=status_var, 
                 anchor=tk.CENTER,       
                 bg="orange",      
                 height=4,              
                 width=17,              
                 bd=3,                  
                 font=("Arial", 14, "bold"),   
                 fg="black",             
                 padx=15,               
                 pady=15,                
                 justify=tk.CENTER,    
                 relief=tk.RAISED,           
                 wraplength=250         
                )
# Pack the label into the window
Status_lb.place(x=800, y=0)

#################################################
# UR5 button - semi placeholder
#################################################

def Camera():
    cam = tk.Toplevel(root) #create a top level root that opens when function called
    cam.title("UR5 Camera")
    cam.geometry("600x600")

    title_label = tk.Label(cam, text="UR5 Camera", font=("Arial", 16), width = 17)
    title_label.place(x=150, y=0)

    # Load and resize image
    image = Image.open("flower.png") #open image of flower in folder on top root
    image = image.resize((500, 400))

    photo = ImageTk.PhotoImage(image)

    img_label = tk.Label(cam, image=photo)
    img_label.image = photo
    img_label.place(x=50, y=50)

    button1 = tk.Button(cam, text="Exit", command=cam.destroy)
    button1.place(x=375, y=0)

  #when button is pressed, calls camera subroutine
    
UR5 = tk.Button(root, 
                   text="UR5_Camera", 
                   command=Camera,
                   activebackground="grey",
                   anchor="center",
                   bd=3,
                   bg="lightgray",
                   cursor="hand2",
                   disabledforeground="red",
                   fg="black",
                   font=("Arial", 14, "bold"),
                   height=4,
                   highlightbackground="grey",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised",
                   padx=10,
                   pady=8,
                   width=15,
                   wraplength=250)

# position the button
UR5.place(x = 1050, y = 0)

#################################################
# Action Log
#################################################

def Generate_Report():
    import datetime
    global action_history
    now = datetime.datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S")
    f = open(f"Logs/{filename}.csv", "w")
    f.write("Time,Event")
    f.write("\n")
    for entry in action_history:
        time_part = entry.split("]")[0].replace("[", "")
        event_part = entry.split("]")[1].strip()
        f.write(f"{time_part},{event_part}\n")
    



def Log():
    top = tk.Toplevel(root)
    top.title("Action Log")
    top.geometry("800x800")

    Log_label = tk.Label(top, text="Action Log", font=("Arial", 16))

    
    Log_label.place(x=350, y=50)
    button2 = tk.Button(top, text="Exit", activebackground="grey",bg="lightgray",font = ("Arial", 16),width = 10, height = 3,command=top.destroy)
    button2.place(x=100, y= 700)


    ##change later to generate a report
    button3 = tk.Button(top, text="Generate Report", activebackground="grey",bg="lightgray",font = ("Arial", 16),width = 15, height = 3,command=Generate_Report)
    button3.place(x=525, y= 700)

    text_area = scrolledtext.ScrolledText(top, 
                                      wrap = tk.WORD, 
                                      width = 50, 
                                      height = 27, 
                                      font = ("Times New Roman",
                                              15))

    text_area.place(y = 75, x = 150)

    # Placing cursor in the text area
    text_area.configure(state="normal")
    for entry in action_history:
        text_area.insert(tk.END, entry + "\n")
    text_area.configure(state="disabled")

Action_Log = tk.Button(root, 
                   text="Action Log", 
                   command=Log,
                   activebackground="grey",
                   anchor="center",
                   bd=3,
                   bg="lightgray",
                   cursor="hand2",
                   disabledforeground="red",
                   fg="black",
                   font=("Arial", 14, "bold"),
                   height=4,
                   highlightbackground="grey",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised",
                   padx=10,
                   pady=8,
                   width=15,
                   wraplength=250)

# position the button
Action_Log.place(x = 1275, y = 0)

#################################################
# Job Queue
#################################################


Queue = scrolledtext.ScrolledText(root, 
                                    wrap = tk.WORD, 
                                    width = 40, 
                                    height = 23, 
                                    font = ("Times New Roman",
                                            15))
Queue.place(x = 1075, y =140)
Queue.tag_configure("underline", underline=True)
Queue.tag_configure("center", justify='center')
Queue.focus()
Queue.insert(tk.INSERT, "Job Queue:\n", ("center", "underline"))
Queue.configure(state ='disabled')


#################################################
# ridgeback communication
#################################################

def on_queue_click(event): # get the line user has clicked
    global selected_plant
    idx = Queue.index(f"@{event.x},{event.y}")
    line = Queue.get(f"{idx} linestart", f"{idx} lineend").strip()

    if line.startswith("Plant "): #if a plant statement is able to be clicked updates selected plant variable
        try:
            selected_plant = int(line.split()[1])
            status_var.set(f"Status: Selected Plant {selected_plant}")
        except:
            selected_plant = None

Queue.bind("<Button-1>", on_queue_click)

def send_to_ridgeback():
    global selected_plant
    if selected_plant is None: #wont run if no plant selected
        status_var.set("Status: Click a plant in the job queue first")
        return

    # Send command to robot to go to certain plant
    if ser:
        ser.write(f"RIDGEBACK,{selected_plant}\n".encode())
        ser.flush()

    # Remove from queue + clear selection
    Queue.configure(state="normal")
    delete_plant_line(selected_plant)
    Queue.configure(state="disabled")

    status_var.set(f"Status: Sent Plant {selected_plant} to Ridgeback")
    log_event(f"User decided to water Plant {selected_plant}")
    selected_plant = None #ensures no wrong actions after sent message

def reject_plant():
    global selected_plant
    if selected_plant is None: #wont run if no plant selected
        status_var.set("Status: Click a plant in the job queue first")
        return

    Queue.configure(state="normal")
    delete_plant_line(selected_plant)
    Queue.configure(state="disabled")

    status_var.set(f"Status: Rejected Plant {selected_plant}")
    set_plant(selected_plant, 2) #sets last status and current status to watered
    last_status[selected_plant] = 2
    log_event(f"User decided to reject Plant {selected_plant}")
    selected_plant = None #ensures no wrong actions after rejected

SendBtn = tk.Button(root, text="Send to Ridgeback", command=send_to_ridgeback,
                    font=("Arial", 14, "bold"), bg="lightgreen", bd=3, width=15, height = 3)
SendBtn.place(x=1080, y=660)

RejectBtn = tk.Button(root, text="Reject Plant", command=reject_plant,
                      font=("Arial", 14, "bold"), bg="lightcoral", bd=3, width=15, height = 3)
RejectBtn.place(x=1280, y=660)

#################################################
# Create area for shape drawing of plants
#################################################
Env = tk.Canvas(root, width=1025, height=650, bg="lightblue", highlightthickness=0)
Env.place(x=25, y=140)

#creates tables for plants to lie on
Row1 = Env.create_rectangle( 
    175, 125, 975, 150,
    fill="white",
    outline="black",
    width=3
)

Row2 = Env.create_rectangle(
    175, 325, 975, 350,
    fill="white",
    outline="black",
    width=3
)

Row3 = Env.create_rectangle(
    175, 525, 975, 550,
    fill="white",
    outline="black",
    width=3
)

#creates plants
plant1 = Env.create_oval(
    190, 85, 240, 135,
    fill="green",
    outline="black",
    width=3
)

plant2 = Env.create_oval(
    290, 85, 340, 135,
    fill="green",
    outline="black",
    width=3
)

plant3 = Env.create_oval(
    390, 85, 440, 135,
    fill="green",
    outline="black",
    width=3
)

plant4 = Env.create_oval(
    490, 85, 540, 135,
    fill="green",
    outline="black",
    width=3
)

plant5 = Env.create_oval(
    590, 85, 640, 135,
    fill="green",
    outline="black",
    width=3
)

plant6 = Env.create_oval(
    690, 85, 740, 135,
    fill="green",
    outline="black",
    width=3
)

plant7 = Env.create_oval(
    790, 85, 840, 135,
    fill="green",
    outline="black",
    width=3
)

plant8 = Env.create_oval(
    890, 85, 940, 135,
    fill="green",
    outline="black",
    width=3
)

plant9 = Env.create_oval(
    235, 135, 285, 185,
    fill = "green",
    outline = "black",
    width =3
)

plant10 = Env.create_oval(
    335, 135, 385, 185,
    fill="green",
    outline="black",
    width=3
)

plant11 = Env.create_oval(
    435, 135, 485, 185,
    fill="green",
    outline="black",
    width=3
)

plant12 = Env.create_oval(
    535, 135, 585, 185,
    fill="green",
    outline="black",
    width=3
)

plant13 = Env.create_oval(
    635, 135, 685, 185,
    fill="green",
    outline="black",
    width=3
)

plant14 = Env.create_oval(
    735, 135, 785, 185,
    fill="green",
    outline="black",
    width=3
)

plant15 = Env.create_oval(
    835, 135, 885, 185,
    fill="green",
    outline="black",
    width=3
)

plant16 = Env.create_oval(
    935, 135, 985, 185,
    fill="green",
    outline="black",
    width=3
)

plant17 = Env.create_oval(
    190, 285, 240, 335,
    fill="green",
    outline="black",
    width=3
)

plant18 = Env.create_oval(
    290, 285, 340, 335,
    fill="green",
    outline="black",
    width=3
)

plant19 = Env.create_oval(
    390, 285, 440, 335,
    fill="green",
    outline="black",
    width=3
)

plant20 = Env.create_oval(
    490, 285, 540, 335,
    fill="green",
    outline="black",
    width=3
)

plant21 = Env.create_oval(
    590, 285, 640, 335,
    fill="green",
    outline="black",
    width=3
)

plant22 = Env.create_oval(
    690, 285, 740, 335,
    fill="green",
    outline="black",
    width=3
)

plant23 = Env.create_oval(
    790, 285, 840, 335,
    fill="green",
    outline="black",
    width=3
)

plant24 = Env.create_oval(
    890, 285, 940, 335,
    fill="green",
    outline="black",
    width=3
)

plant24 = Env.create_oval(
    890, 285, 940, 335,
    fill="green",
    outline="black",
    width=3
)

plant25 = Env.create_oval(
    240, 335, 290, 385,
    fill="green",
    outline="black",
    width=3
)

plant26 = Env.create_oval(
    340, 335, 390, 385,
    fill="green",
    outline="black",
    width=3
)

plant27 = Env.create_oval(
    440, 335, 490, 385,
    fill="green",
    outline="black",
    width=3
)

plant28 = Env.create_oval(
    540, 335, 590, 385,
    fill="green",
    outline="black",
    width=3
)

plant29 = Env.create_oval(
    640, 335, 690, 385,
    fill="green",
    outline="black",
    width=3
)

plant30 = Env.create_oval(
    740, 335, 790, 385,
    fill="green",
    outline="black",
    width=3
)

plant31 = Env.create_oval(
    840, 335, 890, 385,
    fill="green",
    outline="black",
    width=3
)

plant32 = Env.create_oval(
    940, 335, 990, 385,
    fill="green",
    outline="black",
    width=3
)

plant33 = Env.create_oval(
    195, 485, 245, 535,
    fill="green",
    outline="black",
    width=3
)

plant34 = Env.create_oval(
    295, 485, 345, 535,
    fill="green",
    outline="black",
    width=3
)

plant35 = Env.create_oval(
    395, 485, 445, 535,
    fill="green",
    outline="black",
    width=3
)

plant36 = Env.create_oval(
    495, 485, 545, 535,
    fill="green",
    outline="black",
    width=3
)

plant37 = Env.create_oval(
    595, 485, 645, 535,
    fill="green",
    outline="black",
    width=3
)

plant38 = Env.create_oval(
    695, 485, 745, 535,
    fill="green",
    outline="black",
    width=3
)

plant39 = Env.create_oval(
    795, 485, 845, 535,
    fill="green",
    outline="black",
    width=3
)

plant40 = Env.create_oval(
    895, 485, 945, 535,
    fill="green",
    outline="black",
    width=3
)

plant41 = Env.create_oval(
    240, 535, 290, 585,
    fill="green",
    outline="black",
    width=3
)

plant42 = Env.create_oval(
    340, 535, 390, 585,
    fill="green",
    outline="black",
    width=3
)

plant43 = Env.create_oval(
    440, 535, 490, 585,
    fill="green",
    outline="black",
    width=3
)

plant44 = Env.create_oval(
    540, 535, 590, 585,
    fill="green",
    outline="black",
    width=3
)

plant45 = Env.create_oval(
    640, 535, 690, 585,
    fill="green",
    outline="black",
    width=3
)

plant46 = Env.create_oval(
    740, 535, 790, 585,
    fill="green",
    outline="black",
    width=3
)

plant47 = Env.create_oval(
    840, 535, 890, 585,
    fill="green",
    outline="black",
    width=3
)

plant48 = Env.create_oval(
    940, 535, 990, 585,
    fill="green",
    outline="black",
    width=3
)

###########################################
# Robot - placeholder
###########################################
robo_var = tk.StringVar()
robo_var.set("Robot")

# Create the label widget with all options
Robot_lb = tk.Label(root, 
                 textvariable=robo_var, 
                 anchor=tk.CENTER,       
                 bg="black",      
                 height=3,              
                 width=5,              
                 bd=3,                  
                 font=("Arial", 16, "bold"),   
                 fg="white",             
                 padx=15,               
                 pady=15,                
                 justify=tk.CENTER,    
                 relief=tk.RAISED,           
                 wraplength=250         
                )
# Pack the label into the window
Robot_lb.place(x=50, y=650)


###########################################
# Py serial and main loop area
###########################################

# dictionary to store ID values of each plant
plants = {  
    1: plant1, 2: plant2, 3: plant3, 4: plant4,
    5: plant5, 6: plant6, 7: plant7, 8: plant8,
    9: plant9, 10: plant10, 11: plant11, 12: plant12,
    13: plant13, 14: plant14, 15: plant15, 16: plant16,
    17: plant17, 18: plant18, 19: plant19, 20: plant20,
    21: plant21, 22: plant22, 23: plant23, 24: plant24,
    25: plant25, 26: plant26, 27: plant27, 28: plant28,
    29: plant29, 30: plant30, 31: plant31, 32: plant32,
    33: plant33, 34: plant34, 35: plant35, 36: plant36,
    37: plant37, 38: plant38, 39: plant39, 40: plant40,
    41: plant41, 42: plant42, 43: plant43, 44: plant44,
    45: plant45, 46: plant46, 47: plant47, 48: plant48
}

from datetime import datetime

action_history = []  # stores all log messages

# Action Log Updating
#########################################################

def log_event(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}" #appends action_history by first placing a timestamp and a message
    action_history.append(entry)

def delete_plant_line(plant_id):
    pattern = f"Plant {plant_id} is " #finds pattern in messages to identify line
    start = "1.0"

    while True:
        idx = Queue.search(pattern, start, tk.END)#search for that pattern
        if not idx:
            break

        line_start = f"{idx} linestart"
        line_end = f"{idx} lineend+1c"
        Queue.delete(line_start, line_end)#removes line from job queue

        start = line_start

# Plant colour setting
#######################################################

def set_plant(plant_id, status):
    match status:
        case 1:
            colour = "yellow" #underwatered
        case 2:
            colour = "green" # watered
        case 3:
            colour = "red"  #overwatered
        case _:
            colour = "grey" #disconnected
    if plant_id in plants:
        Env.itemconfig(plants[plant_id], fill=colour)

        

def remove_from_queue_if_present(plant_id: int):
    # remove job-queue line if it exists
    Queue.configure(state="normal")
    delete_plant_line(plant_id)
    Queue.configure(state="disabled")

    # if user had it selected, clear selection
    global selected_plant
    if selected_plant == plant_id:
        selected_plant = None
        status_var.set("Status: Selected plant went offline")

# Serial setup
PORT = "COM6"
BAUD = 9600
last_status = {}#stores last plant so not spammed in job queue
queue_positions = {}
selected_plant = None
action_history = []  # stores all log messages



from datetime import datetime

ser = None

last_seen = {}
offline_plants = set()
OFFLINE_TIMEOUT_S = 5          # if not heard from after 5 seconds will go grey
STARTUP_GRACE_S = 2            # grace period before greying never-seen plants
start_time = datetime.now()

# Make all plants start grey (unknown) until they report
for pid in plants.keys():
    set_plant(pid, -1)         # -1 will map to default and set them all grey


# Serial setup
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    status_var.set("Status: Connected")
except Exception:
    ser = None
    status_var.set("Status: No Serial")


def poll_serial():
    now = datetime.now()


    for plant_id in plants.keys(): #checks if online 
        seen = last_seen.get(plant_id, None)

        # If never seen, treat as offline after startup grace
        if seen is None:
            if (now - start_time).total_seconds() > STARTUP_GRACE_S: # if most recent time at sampling time - start time is greater than 5 seconds, state it as disconnected
                if plant_id not in offline_plants:
                    remove_from_queue_if_present(plant_id)
                    offline_plants.add(plant_id) 
                    set_plant(plant_id, -1)  # set colour of plant grey
                    #offline action
                    log_event(f"Plant Node {plant_id} is offline")
            continue

        # If seen before and now no longer receiving messages
        if (now - seen).total_seconds() > OFFLINE_TIMEOUT_S: #if greater than 5, set it grey
            if plant_id not in offline_plants:
                remove_from_queue_if_present(plant_id)
                offline_plants.add(plant_id)
                set_plant(plant_id, -1)  # grey
                log_event(f"Plant {plant_id} offline (no data for {OFFLINE_TIMEOUT_S}s)")

   # Serial reading
   ##########################################
    if ser is not None and ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        print(line) #receives incoming Lora request to water each plant

        if "," in line:
            a, b = line.split(",", 1)
            try:
                plant_id = int(a)
                status = int(b)

                
                last_seen[plant_id] = datetime.now() #updates timestamp of when last interacted with

                # bring back online if previously offline
                if plant_id in offline_plants:
                    offline_plants.remove(plant_id) #if now online again, mark as now online
                    log_event(f"Plant {plant_id} back online")

                set_plant(plant_id, status) #sets status back from grey to actual

                if last_status.get(plant_id) != status: #ensures same messages to the job queue dont appear again if they have the same status as before, only when status has changed
                    last_status[plant_id] = status

                    Queue.configure(state="normal")
                    delete_plant_line(plant_id) #delete old line from this plant

                    if status == 1: #sends message to action log and job queue accordingly 
                        Queue.insert(tk.END, f"Plant {plant_id} is UNDERWATERED\n")
                        log_event(f"Plant {plant_id} reported UNDERWATERED")
                    elif status == 3:
                        Queue.insert(tk.END, f"Plant {plant_id} is OVERWATERED, intervention needed\n")
                        log_event(f"Plant {plant_id} reported OVERWATERED")
                    else:
                        log_event(f"Plant {plant_id} reported WATERED")

                    Queue.configure(state="disabled")

            except ValueError:
                pass

    root.after(50, poll_serial) #will re run this subroutine again in 50 ms



poll_serial()



# Run the main event loop
root.mainloop()

