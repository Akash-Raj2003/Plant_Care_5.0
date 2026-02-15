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
root.title("Welcome to My App")  # Set window title

import tkinter as tk

#################################################
# Stop Button - semi placeholder
##################################################
def Stop():
    print("Stop")

# Creating a button with specified options
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
                   font=("Arial", 16, "bold"),
                   height=3,
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
                 height=3,              
                 width=15,              
                 bd=3,                  
                 font=("Arial", 16, "bold"),   
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
                 height=3,              
                 width=15,              
                 bd=3,                  
                 font=("Arial", 16, "bold"),   
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
# Status Label - Placeholder
#################################################
status_var = tk.StringVar()
status_var.set("Status: Idle")

# Create the label widget with all options
Status_lb = tk.Label(root, 
                 textvariable=status_var, 
                 anchor=tk.CENTER,       
                 bg="orange",      
                 height=3,              
                 width=15,              
                 bd=3,                  
                 font=("Arial", 16, "bold"),   
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
    cam = tk.Toplevel(root)
    cam.title("UR5 Camera")
    cam.geometry("600x600")

    title_label = tk.Label(cam, text="UR5 Camera", font=("Arial", 16))
    title_label.place(x=150, y=0)

    # Load and resize image
    image = Image.open("flower.png")
    image = image.resize((500, 400))

    photo = ImageTk.PhotoImage(image)

    img_label = tk.Label(cam, image=photo)
    img_label.image = photo
    img_label.place(x=50, y=50)

    button1 = tk.Button(cam, text="Exit", command=cam.destroy)
    button1.place(x=375, y=0)

  
    
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
                   font=("Arial", 16, "bold"),
                   height=3,
                   highlightbackground="grey",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised",
                   padx=10,
                   pady=8,
                   width=13,
                   wraplength=250)

# position the button
UR5.place(x = 1050, y = 0)

#################################################
# Action Log - semi placeholder
#################################################

def Log():
    top = tk.Toplevel(root)
    top.title("Action Log")
    top.geometry("600x600")

    Log_label = tk.Label(top, text="Action Log", font=("Arial", 16))

    
    Log_label.place(x=150, y=0)
    button2 = tk.Button(top, text="Exit", command=top.destroy)
    button2.place(x=375, y=0)

    text_area = scrolledtext.ScrolledText(top, 
                                      wrap = tk.WORD, 
                                      width = 50, 
                                      height = 10, 
                                      font = ("Times New Roman",
                                              15))

    text_area.place(y = 25, x = 0)

    # Placing cursor in the text area
    text_area.focus()
    text_area.insert(tk.INSERT,
    """\
    Activity 1
    Activity 2
    Activity 3
    """)
    text_area.configure(state ='disabled')

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
                   font=("Arial", 16, "bold"),
                   height=3,
                   highlightbackground="grey",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised",
                   padx=10,
                   pady=8,
                   width=13,
                   wraplength=250)

# position the button
Action_Log.place(x = 1275, y = 0)

#################################################
# Job Queue - placeholder
#################################################


Queue = scrolledtext.ScrolledText(root, 
                                    wrap = tk.WORD, 
                                    width = 40, 
                                    height = 29, 
                                    font = ("Times New Roman",
                                            15))
Queue.place(x = 1075, y =140)
Queue.tag_configure("underline", underline=True)
Queue.tag_configure("center", justify='center')
Queue.focus()
Queue.insert(tk.INSERT, "Job Queue:\n", ("center", "underline"))
Queue.insert(tk.INSERT,
"""Activity 1
Activity 2
Activity 3
""", "center")
Queue.configure(state ='disabled')

#################################################
# Create area for shape drawing of plants
#################################################
Env = tk.Canvas(root, width=1025, height=650, bg="lightblue", highlightthickness=0)
Env.place(x=25, y=140)

# --- Rectangle inside the blue area ---
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

# --- Rectangle inside the blue area ---
Row3 = Env.create_rectangle(
    175, 525, 975, 550,
    fill="white",
    outline="black",
    width=3
)

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
# Py serial to update colours
###########################################

plants = { #dictionary to store ID values of each plant
    1: plant1, 2: plant2, 3: plant3, 4: plant4,
    5: plant5, 6: plant6, 7: plant7, 8: plant8,
    9: plant9, 10: plant10, 11: plant11, 12: plant12,
    13: plant13, 14: plant14, 15: plant15, 16: plant16
}

def set_plant(plant_id, status):
    match status:
        case 1:
            colour = "yellow"
        case 2:
            colour = "green"
        case 3:
            colour = "red"
        case _:
            colour = "grey"
    if plant_id in plants:
        Env.itemconfig(plants[plant_id], fill=colour)


# Serial setup
PORT = "COM6"
BAUD = 9600

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    status_var.set("Status: Connected")
except Exception as e:
    ser = None
    status_var.set("Status: No Serial")

def poll_serial():
    # This function runs again and again during mainloop
    if ser and ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if "," in line:
            a, b = line.split(",", 1)
            try:
                plant_id = int(a)      # "001" -> 1
                status = int(b)        # "1" -> 1
                set_plant(plant_id, status)
            except ValueError:
                pass  # ignore bad lines
    root.after(50, poll_serial)  # run again in 50 ms

poll_serial()



# Run the main event loop
root.mainloop()

