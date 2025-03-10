import cv2
import os
import numpy as np
import time
import csv
import shutil
import threading
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tkinter import Button, Entry, Label, messagebox, Toplevel, Listbox, SINGLE, Frame, Scrollbar

# Directories and file paths
DATA_DIR = "face_data"  # Directory to store face images
ATTENDANCE_LOG = "attendance_log.csv"  # CSV file to store attendance records
UNKNOWN_FACE_DIR = "unknown_faces"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Create the face data directory if it doesn't exist

if not os.path.exists(ATTENDANCE_LOG):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_LOG, index=False)  # Create an empty attendance log

if not os.path.exists(UNKNOWN_FACE_DIR):
    os.makedirs(UNKNOWN_FACE_DIR)  # Create the folder if it doesn't exist

# Load Haar cascade for face detection
if not os.path.exists('haar_face.xml'):
    messagebox.showerror("Error", "haar_face.xml not found! Ensure the file is in the correct directory.")
    exit()
face_cascade = cv2.CascadeClassifier('haar_face.xml')

# Load LBPH face recognizer
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Try the first method
except AttributeError:
    try:
        recognizer = cv2.face_LBPHFaceRecognizer.create()  # Fallback method
    except AttributeError:
        messagebox.showerror("Error", "LBPHFaceRecognizer not available in your OpenCV version. Please install opencv-contrib-python.")
        exit()

# Initialize KNN model for facial recognition
knn_model = KNeighborsClassifier(n_neighbors=3)  # Using 3 neighbors for KNN classification
knn_trained = False  # Flag to track if KNN model is trained
label_encoder = LabelEncoder()  # Convert labels (names) into numerical values
scaler = StandardScaler()  # Normalize feature data for KNN


# Function to create scrolling text effect
def scroll_text():
    global text_index
    if text_index < len(full_text):
        scrolling_label.config(text=full_text[:text_index])
        text_index += 1
        root.after(150, scroll_text)  # Adjust speed here
    else:
        root.after(1500, reset_text)  # Hold full text before restarting

def reset_text():
    global text_index
    text_index = 0
    scrolling_label.config(text="")
    root.after(500, scroll_text)

# Function to show text description on hover
def show_description(event, text):
    description_label.config(text=text)

def clear_description(event):
    description_label.config(text="")

# Function for smooth color transition
def smooth_transition(widget, start_color, end_color, step=0, max_steps=15):
    if step <= max_steps:
        r1, g1, b1 = root.winfo_rgb(start_color)
        r2, g2, b2 = root.winfo_rgb(end_color)

        # Interpolating color transition
        new_r = int(r1 + (r2 - r1) * (step / max_steps)) // 256
        new_g = int(g1 + (g2 - g1) * (step / max_steps)) // 256
        new_b = int(b1 + (b2 - b1) * (step / max_steps)) // 256

        new_color = f"#{new_r:02x}{new_g:02x}{new_b:02x}"
        widget.config(bg=new_color)
        root.after(20, lambda: smooth_transition(widget, start_color, end_color, step + 1, max_steps))

# Function to update status bar
def update_status(action):
    status_label.config(text=f"Status: {action}")

# Function to check if new data is available for retraining
def is_new_data_available():
    return any(os.listdir(os.path.join(DATA_DIR, user)) for user in os.listdir(DATA_DIR))

# Global variable for label mapping
label_map = {}

# Function to train the face recognizer
def train_recognizer():
    """Train LBPH and KNN recognizers with incremental training."""
    global label_map, knn_trained
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, knn_data, knn_labels = [], [], [], []
    label_map = {}

    try:
        # Load existing model if available
        if os.path.exists("trained_model.yml"):
            recognizer.read("trained_model.yml")
            print("Existing model loaded successfully.")
        else:
            print("No existing model found, starting fresh.")
    except cv2.error as e:
        print(f"Error loading trained_model.yml: {e}")
        recognizer = cv2.face.LBPHFaceRecognizer_create()

    for label, name in enumerate(os.listdir(DATA_DIR)):
        user_folder = os.path.join(DATA_DIR, name)
        if not os.path.isdir(user_folder):
            continue

        label_map[label] = name  # Store mapping of label to name

        for img_file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (200, 200))
            img = cv2.equalizeHist(img)
            img_flattened = img.flatten()  # Flatten the image to a 1D array

            faces.append(img)
            labels.append(label)
            knn_data.append(img_flattened)
            knn_labels.append(name)

    # Train LBPH model
    if faces:
        if os.path.exists("trained_model.yml"):
            recognizer.update(faces, np.array(labels))  # Incremental update
        else:
            recognizer.train(faces, np.array(labels))  # Train from scratch
        recognizer.save("trained_model.yml")

    # Train KNN model
    if knn_data:
        knn_data = scaler.fit_transform(knn_data)
        knn_model.fit(knn_data, label_encoder.fit_transform(knn_labels))
        knn_trained = True

    return is_new_data_available()

# Function to retrain the model manually
def retrain_model():
    """Manually retrain the model with incremental training."""
    update_status("Retrainingg....")
    new_data_available = train_recognizer()
    if new_data_available:
        update_status("Model updated with new data!")
        messagebox.showinfo("Info", "Model updated with new data!")
    else:
        update_status("No new data found for training.")
        messagebox.showinfo("Info", "No new data found for training.")
    update_status("Ready")

# Ensure initial training
train_recognizer()

def register_user():
    def capture_faces():
        update_status("Capturing Faces...")
        name = entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Name cannot be empty.")
            return

        user_folder = os.path.join(DATA_DIR, name)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        cap = cv2.VideoCapture(0)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # Enhances contrast
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                count += 1
                cv2.imwrite(f"{user_folder}/{count}.jpg", face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Register User", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= 200:
                break

        cap.release()
        cv2.destroyAllWindows()

        train_recognizer()
        update_status(f"User {name} registered successfully!")
        messagebox.showinfo("Info", f"User {name} registered successfully!")
        popup.destroy()

    update_status("Registering User...")
    popup = Toplevel(root)
    popup.title("Register New User")
    Label(popup, text="Enter Name:").pack()
    entry = Entry(popup)
    entry.pack()
    Button(popup, text="Capture Faces", command=capture_faces).pack()
    update_status("Ready")

def real_time_tracking():
    """Track, recognize, and auto-learn unrecognized faces."""
    update_status("Starting Real-Time Face Tracking...")
    global knn_trained

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access webcam!")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Enhances contrast
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            # Resize the face to a fixed size without distorting the aspect ratio
            face_resized = cv2.resize(face, (200, 200), interpolation=cv2.INTER_LINEAR)

            # Recognize face using KNN or LBPH
            if knn_trained:
                face_flattened = face_resized.reshape(1, -1)  # Ensure it's 2D (1 sample, n features)
                pred_label = knn_model.predict(face_flattened)[0]  # Now properly formatted
                name = label_encoder.inverse_transform([pred_label])[0]
            else:
                label, confidence = recognizer.predict(face_resized)
                name = label_map.get(label, "Unknown")

            # Draw bounding box and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Auto-save unknown faces
            if name == "Unknown":
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                unknown_face_path = os.path.join(UNKNOWN_FACE_DIR, f"unknown_{timestamp}.jpg")
                cv2.imwrite(unknown_face_path, face_resized)

        cv2.imshow("Real-Time Face Tracking & Auto-Learning", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    update_status("Ready")

# Run in a separate thread to prevent GUI lag
def start_real_time_tracking_with_auto_learning():
    tracking_thread = threading.Thread(target=real_time_tracking, daemon=True)
    tracking_thread.start()

def recognize_face():
    """Recognizes a face and marks attendance if not already recorded."""
    update_status("Recognizing Face...")
    if os.stat(ATTENDANCE_LOG).st_size == 0:
        messagebox.showwarning("Warning", "Attendance log is empty. No records to process.")
        return

    attendance_df = pd.read_csv(ATTENDANCE_LOG)
    today = datetime.now().strftime("%Y-%m-%d")

    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Info", "Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Enhances contrast
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            # Resize the face to a fixed size without distorting the aspect ratio
            face_resized = cv2.resize(face, (200, 200), interpolation=cv2.INTER_LINEAR)


            # Recognize face using KNN or LBPH
            if knn_trained:
                face_flattened = face_resized.reshape(1, -1)  # Ensure it's 2D (1 sample, n features)
                pred_label = knn_model.predict(face_flattened)[0]  # Now properly formatted
                name = label_encoder.inverse_transform([pred_label])[0]
            else:
                label, confidence = recognizer.predict(face_resized)
                name = label_map.get(label, "Unknown")

            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if name != "Unknown":
                # Check if attendance is already marked
                if not attendance_df[(attendance_df["Name"] == name) & (attendance_df["Date"] == today)].empty:
                    messagebox.showinfo("Info", f"Attendance already marked for {name} today.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                # Mark attendance
                timestamp = datetime.now().strftime("%H:%M:%S")
                with open(ATTENDANCE_LOG, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, today, timestamp])
                messagebox.showinfo("Success", f"Attendance marked for {name}.")
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow("Recognize Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    update_status("Ready")
    cap.release()
    cv2.destroyAllWindows()

def view_registered_users():
    update_status("Viewing Registered Users...")
    users = os.listdir(DATA_DIR)
    # Create a new window to display registered users
    users_window = Toplevel(root)
    users_window.title("Registered Users")
    users_window.geometry("350x400")
    Label(users_window, text="Registered Users", font=("Arial", 12, "bold")).pack(pady=5)
    frame = Frame(users_window)
    frame.pack(fill="both", expand=True)

    scrollbar = Scrollbar(frame)
    listbox = Listbox(frame, selectmode=SINGLE, yscrollcommand=scrollbar.set, width=40, height=15)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)
    i = 1
    for user in users:
        listbox.insert("end", f"{i}. {user}")
        i += 1

    Button(users_window, text="Close", command=users_window.destroy).pack(pady=5)
    update_status("Ready")

def show_attendance():
    """Displays the attendance of students for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    update_status("Showing Attendance...")
    # Read the attendance log
    if not os.path.exists(ATTENDANCE_LOG) or os.stat(ATTENDANCE_LOG).st_size == 0:
        messagebox.showwarning("Warning", "No attendance records found.")
        return

    attendance_df = pd.read_csv(ATTENDANCE_LOG)
    today_attendance = attendance_df[attendance_df["Date"] == today]

    if today_attendance.empty:
        messagebox.showinfo("Info", "No attendance has been marked today.")
        return

    # Create a new window to display attendance
    attendance_window = Toplevel(root)
    attendance_window.title("Today's Attendance")
    attendance_window.geometry("350x400")

    Label(attendance_window, text="Today's Attendance", font=("Arial", 12, "bold")).pack(pady=5)

    frame = Frame(attendance_window)
    frame.pack(fill="both", expand=True)

    scrollbar = Scrollbar(frame)
    listbox = Listbox(frame, selectmode=SINGLE, yscrollcommand=scrollbar.set, width=40, height=15)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    for _, row in today_attendance.iterrows():
        listbox.insert("end", f"{row['Name']} - {row['Time']}")

    Button(attendance_window, text="Close", command=attendance_window.destroy).pack(pady=5)
    update_status("Ready")

def delete_user():
    def perform_deletion():
        try:
            selected_user = listbox.get(listbox.curselection())
            if not selected_user:
                messagebox.showerror("Error", "No user selected.")
                return
            user_folder = os.path.join(DATA_DIR, selected_user)
            # Recursively delete the user folder and its contents
            shutil.rmtree(user_folder)
            train_recognizer()  # Force retraining after deletion
            update_status(f"User {selected_user} deleted successfully!")
            messagebox.showinfo("Info", f"User {selected_user} deleted successfully!")
            popup.destroy()
        except PermissionError as e:
            messagebox.showerror("Permission Error", f"Error deleting user: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"First Select the User and then click the button")

    update_status("Deleting User...")
    popup = Toplevel(root)
    popup.title("Delete User")

    Label(popup, text="Select a user to delete:").pack()

    listbox = Listbox(popup, selectmode=SINGLE)
    for user in os.listdir(DATA_DIR):
        listbox.insert("end", user)
    listbox.pack()

    Button(popup, text="Delete", command=perform_deletion).pack()
    update_status("Ready")


def exit_app(event=None):
    root.quit()

# Initialize GUI
root = tk.Tk()
root.title("Facial Attendance Monitoring System")
root.attributes('-fullscreen', True)  # Full-screen mode
root.configure(bg="white")
root.bind("<Escape>", exit_app)  # Exit on ESC key

# Load and display logo
logo_path = "logo.png"
if os.path.exists(logo_path):
    logo_img = Image.open(logo_path).resize((root.winfo_screenwidth(), 150), Image.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = Label(root, image=logo_photo, bg="white")
    logo_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

# Scrolling Text (Added Extra Padding at Top)
full_text = " FACIAL ATTENDANCE MONITORING SYSTEM "
text_index = 0
scrolling_label = Label(root, text="", font=("Arial", 28, "bold"), bg="white", fg="red")
scrolling_label.grid(row=1, column=0, columnspan=2, pady=(30, 10))  # Added more top padding
scroll_text()

# Enlarged Status Bar
status_label = Label(root, text="Status: Ready", anchor="w", bg="light gray", font=("Arial", 15, "bold"), height=2)
status_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

# Buttons Layout
buttons = [
    ("Register User", register_user, "Register a new user for attendance system"),
    ("Recognize Face", recognize_face, "Recognize and mark attendance"),
    ("Real-Time Face Tracking", real_time_tracking, "Track faces in real-time"),
    ("Show Attendance", show_attendance, "View attendance logs"),
    ("View Registered Users", view_registered_users, "List all registered users"),
    ("Delete User", delete_user, "Remove a user from the system"),
    ("Retrain Model", retrain_model, "Retrain face recognition model"),
    ("Exit", exit_app, "Exit the application")
]

button_frame = Frame(root, bg="white")
button_frame.grid(row=3, column=0, columnspan=2, pady=20)

# Creating buttons in 2 columns with smooth hover effect
for i, (text, command, desc) in enumerate(buttons):
    row, col = divmod(i, 2)
    btn = Button(
        button_frame, text=text, command=command,
        width=25, height=2, font=("Arial", 12, "bold"), bg="#3498db", fg="white",
        relief="flat", bd=5, activebackground="#1b4f72", borderwidth=3
    )
    btn.grid(row=row, column=col, padx=10, pady=5, ipadx=5, ipady=2)

    # Properly capture `desc` in lambda function
    btn.bind("<Enter>", lambda event, widget=btn: (
        widget.config(cursor="hand2"),
        smooth_transition(widget, "#3498db", "#1b4f72"),
        show_description(event, desc)
    ))

    btn.bind("<Leave>", lambda event, widget=btn: (
        widget.config(cursor=""),
        smooth_transition(widget, "#1b4f72", "#3498db"),
        clear_description(event)
    ))

# Text description label
description_label = Label(root, text="Description", font=("Arial", 16, "italic"), bg="white", fg="black", height=2)
description_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()