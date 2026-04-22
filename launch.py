#!/usr/bin/env python3
"""
Neural Fraud Detector v2 - GUI Launcher

Cross-platform GUI app that:
- Downloads repo on first run (if not present)
- Launches dashboard in browser
- No terminal commands needed
"""

import os
import sys
import subprocess
import webbrowser
import threading
import time
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox
    from tkinter import filedialog
except ImportError:
    # Fallback if tkinter not available
    tk = None

# Configuration
REPO_URL = "https://github.com/codezeroexe/neural-fraud-detector-v2.git"
REPO_NAME = "neural-fraud-detector-v2"
APP_PORT = 5000
APP_URL = f"http://127.0.0.1:{APP_PORT}"
VENV_PATH = "venv"

TRAIN_FILE = "fraudTrain.csv"
MODEL_FILE = "fraud_model.keras"


def get_project_dir():
    """Get the project directory (where this script is)."""
    return Path(__file__).parent.resolve()


def check_files():
    """Check if required files exist."""
    project_dir = get_project_dir()
    train_exists = (project_dir / TRAIN_FILE).exists()
    model_exists = (project_dir / MODEL_FILE).exists()
    return train_exists and model_exists


def open_browser():
    """Open dashboard in browser."""
    time.sleep(1)
    webbrowser.open(APP_URL)


def find_python():
    """Find best Python executable."""
    # Check venv first
    project_dir = get_project_dir()
    venv_python = project_dir / VENV_PATH / "bin" / "python"
    
    if sys.platform == "win32":
        venv_python = project_dir / VENV_PATH / "Scripts" / "python.exe"
    
    if venv_python.exists():
        return str(venv_python)
    
    # Fallback to system python
    return sys.executable


def run_flask():
    """Run Flask app."""
    project_dir = get_project_dir()
    python = find_python()
    
    subprocess.run(
        [python, "app.py"],
        cwd=str(project_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def gui_launch():
    """GUI-based launcher."""
    if tk is None:
        print("tkinter not available. Using terminal mode.")
        terminal_fallback()
        return
    
    project_dir = get_project_dir()
    os.chdir(project_dir)
    
    # Create window
    root = tk.Tk()
    root.title("Neural Fraud Detector v2")
    root.geometry("400x300")
    root.resizable(False, False)
    
    # Center window
    root.eval('tk::PlaceWindow . center')
    
    # Status label
    status_var = tk.StringVar(value="Checking files...")
    status_label = tk.Label(root, textvariable=status_var, font=("Arial", 12), pady=20)
    status_label.pack()
    
    # Progress
    progress_var = tk.StringVar(value="")
    progress_label = tk.Label(root, textvariable=progress_var, font=("Arial", 10), fg="gray")
    progress_label.pack()
    
    # Icon (shieldemoji)
    icon_label = tk.Label(root, text="🛡️", font=("Arial", 48))
    icon_label.pack(pady=10)
    
    root.update()
    
    # Check files
    if check_files():
        status_var.set("Files found!")
        progress_var.set("Launching dashboard...")
        root.update()
        
        # Open browser in background
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run Flask
        threading.Thread(target=run_flask, daemon=True).start()
        
        status_var.set("Dashboard ready!")
        progress_var.set(APP_URL)
        
        # Show info
        messagebox.showinfo(
            "Neural Fraud Detector v2",
            f"Dashboard is running!\n\nOpen: {APP_URL}"
        )
    else:
        # Files not found
        status_var.set("Files not found")
        progress_var.set("Click OK to select folder with data files")
        
        result = messagebox.askyesno(
            "Setup Required",
            "Data files not found.\n\n"
            "Download dataset from:\n"
            "https://www.kaggle.com/datasets/kartik2112/fraud-detection\n\n"
            "Place fraudTrain.csv in project folder,\n"
            "then click Yes to try again.\n\n"
            "Click No to exit."
        )
        
        if result:
            # Try again
            check_and_launch()
        else:
            root.destroy()
            return
    
    # Keep window open briefly
    root.after(3000, root.destroy)
    root.mainloop()


def terminal_fallback():
    """Terminal-based fallback if tkinter fails."""
    project_dir = get_project_dir()
    os.chdir(project_dir)
    
    if check_files():
        print("✓ Files found")
        print(f"✓ Opening {APP_URL}")
        webbrowser.open(APP_URL)
        print(f"✓ Running app.py...")
        python = find_python()
        subprocess.run([python, "app.py"])
    else:
        print("✗ Data files not found")
        print(f"\nPlease:")
        print(f"1. Download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection")
        print(f"2. Place fraudTrain.csv in: {project_dir}")
        print(f"3. Run this again")


def check_and_launch():
    """Check files and launch."""
    if check_files():
        project_dir = get_project_dir()
        os.chdir(project_dir)
        
        threading.Thread(target=open_browser, daemon=True).start()
        threading.Thread(target=run_flask, daemon=True).start()
        
        messagebox.showinfo("Ready", f"Dashboard: {APP_URL}")
    else:
        gui_launch()


# Entry point
if __name__ == "__main__":
    try:
        gui_launch()
    except Exception as e:
        print(f"Error: {e}")
        terminal_fallback()