from tkinter import *
from tkinter import messagebox
import ttkthemes
from PIL import Image, ImageTk
import hashlib

# Secure credentials (Use a database for production)
VALID_CREDENTIALS = {'FourForces': hashlib.sha256('21131'.encode()).hexdigest()}

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def login():
    """Authenticate user credentials."""
    username = userName.get().strip()
    pwd = password.get().strip()

    if not username or not pwd:
        messagebox.showerror('Error', 'Fields cannot be empty!')
        return

    if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == hash_password(pwd):
        messagebox.showinfo('Success', f'Welcome {username}!')
        window.destroy()
        import FAMS  # Ensure FAMS.py exists
    else:
        messagebox.showerror('Error', 'Incorrect Credentials')

def show_password():
    """Toggle password visibility."""
    password.config(show='' if val.get() else '*')

def on_hover(event):
    """Change button color to orange on hover"""
    loginButton.config(bg="#FF9800")

def on_leave(event):
    """Revert button color to green when hover is removed"""
    loginButton.config(bg="#4CAF50")

def animate_button():
    """Click animation"""
    loginButton.config(bg="#E65100")  # Darker Orange
    window.after(200, lambda: loginButton.config(bg="#FF9800"))  # Restore hover color

def exit_fullscreen(event=None):
    """Exit the application on ESC key press."""
    window.quit()

# Initialize Tkinter Window
window = ttkthemes.ThemedTk()
window.set_theme('radiance')

# Fullscreen & ESC to Exit
window.attributes('-fullscreen', True)
window.bind('<Escape>', exit_fullscreen)
window.title('FAMS Login System')

# Load Background Image
bg_color = "#01C5D5"
try:
    bg_image = Image.open("bg.jpg")
    bg_image = bg_image.resize((window.winfo_screenwidth(), window.winfo_screenheight()), Image.Resampling.LANCZOS)
    bgImage = ImageTk.PhotoImage(bg_image)
    bgLabel = Label(window, image=bgImage)
    bgLabel.place(relwidth=1, relheight=1)
except Exception:
    window.configure(bg=bg_color)

team = Label(window, text="FourForces", font=("Times new roman", 30, "bold"), fg="red", bd=0, highlightthickness=0)
team.place(relx=0.09, rely=0.05, anchor=CENTER)  # Positioned in the center top

# Login Frame (Shifted Left)
loginFrame = Frame(window, bg=bg_color, bd=5, relief=RIDGE)
loginFrame.place(relx=0.3, rely=0.5, anchor=CENTER, width=400, height=450)

# Title Inside Login Frame
Label(loginFrame, text="FACIAL \nATTENDANCE\n MONITORING SYSTEM", font=("Times new roman", 20, "bold"), fg="red",bg=bg_color).grid(row=0, column=0, columnspan=2, pady=20)

# Username
Label(loginFrame, text="Username", font=("Goudy old style", 16, "bold"), fg="black", bg=bg_color).grid(row=1, column=0, pady=10, padx=20, sticky="w")
userName = Entry(loginFrame, font=("times new roman", 14), bd=3, bg="#E7E6E6")
userName.grid(row=1, column=1, pady=10, padx=20, ipadx=10)

# Password
Label(loginFrame, text="Password", font=("Goudy old style", 16, "bold"), fg="black", bg=bg_color).grid(row=2, column=0, pady=10, padx=20, sticky="w")
password = Entry(loginFrame, show='*', font=("times new roman", 14), bd=3, bg="#E7E6E6")
password.grid(row=2, column=1, pady=10, padx=20, ipadx=10)

# Show Password Checkbox (Same BG as Login Frame)
val = IntVar()
showPassCheck = Checkbutton(loginFrame, text='Show Password', variable=val, onvalue=1, offvalue=0, command=show_password,
                            bg=bg_color, font=("Arial", 12), fg="black", selectcolor=bg_color, activebackground=bg_color)
showPassCheck.grid(row=3, column=1, pady=5, sticky="w")

# Custom Animated Button
loginButton = Button(
    loginFrame,
    text="Login",
    font=("Arial", 14, "bold"),
    fg="white",
    bg="#4CAF50",  # Default Green
    activebackground="#45a049",
    activeforeground="white",
    cursor="hand2",
    bd=0,
    padx=20,
    pady=5,
    width=15,
    relief="flat",
    command=lambda: [animate_button(), login()]
)

# Bind Hover Effects
loginButton.bind("<Enter>", on_hover)
loginButton.bind("<Leave>", on_leave)

# Place Button
loginButton.grid(row=4, column=0, columnspan=2, pady=20)

# Run the application
window.mainloop()