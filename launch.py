#!/usr/bin/env python3
"""
Neural Fraud Detector v2 - Downloader & Launcher

First run: Downloads repo, installs dependencies, trains model, launches dashboard
Subsequent runs: Detects existing files and launches dashboard directly
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

# Configuration
REPO_URL = "https://github.com/codezeroexe/neural-fraud-detector-v2.git"
REPO_NAME = "neural-fraud-detector-v2"
APP_PORT = 5000
APP_URL = f"http://127.0.0.1:{APP_PORT}"
VENV_PATH = "venv"

# Files to check
TRAIN_FILE = "fraudTrain.csv"
MODEL_FILE = "fraud_model.keras"


def is_downloaded():
    """Check if project files are already downloaded."""
    train_path = Path(TRAIN_FILE)
    model_path = Path(MODEL_FILE)
    
    return train_path.exists() and model_path.exists()


def get_venv_python():
    """Get path to venv Python interpreter."""
    venv_dir = Path(VENV_PATH)
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def get_python():
    """Get Python executable (venv if exists, otherwise system)."""
    if Path(VENV_PATH).exists() and Path(VENV_PATH).is_dir():
        venv_python = get_venv_python()
        if venv_python.exists():
            return str(venv_python)
    return "python3"


def run_command(cmd, cwd=None, check=True):
    """Run shell command."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def print_success(msg):
    print(f"✓ {msg}")


def print_error(msg):
    print(f"✗ {msg}")


def print_info(msg):
    print(f"ℹ {msg}")


def main():
    print("\n" + "=" * 50)
    print("Neural Fraud Detector v2 - Launcher")
    print("=" * 50)
    
    # Get project directory
    project_dir = Path(__file__).parent.resolve()
    os.chdir(project_dir)
    
    # Check if setup
    if is_downloaded():
        print_info("Project files detected")
    else:
        print_info(f"Downloading from {REPO_URL}")
        
        # Clone repository
        parent_dir = project_dir.parent
        clone_path = parent_dir / REPO_NAME
        
        if clone_path.exists():
            print_info("Repository already exists")
        else:
            print_info("Cloning repository...")
            if not run_command(f"git clone {REPO_URL}", cwd=parent_dir):
                print_error("Failed to clone")
                return
            print_success("Cloned")
        
        os.chdir(clone_path)
        project_dir = clone_path
        
        # Set up venv
        print_info("Setting up virtual environment...")
        if not (clone_path / VENV_PATH).exists():
            subprocess.run(f"python3 -m venv {VENV_PATH}", shell=True, cwd=clone_path)
        print_success("Done")
        
        # Install deps
        print_info("Installing dependencies...")
        python = get_venv_python()
        subprocess.run(f'"{python}" -m pip install -r requirements.txt --quiet', 
                   shell=True, cwd=clone_path)
        print_success("Installed")
        
        # Check data
        if not (clone_path / TRAIN_FILE).exists():
            print("\nPlease download dataset from:")
            print("https://www.kaggle.com/datasets/kartik2112/fraud-detection")
            print(f"\nPlace in {clone_path}: fraudTrain.csv, fraudTest.csv")
            print("Then run launcher again.")
            # Still try to launch
    
    # Launch app
    print_info("Starting Flask dashboard...")
    print_success("Opening in browser")
    webbrowser.open(APP_URL)
    
    python = get_python()
    
    print(f"\n{'=' * 50}")
    print(f"Dashboard: {APP_URL}")
    print(f"Press Ctrl+C to stop")
    print(f"{'=' * 50}\n")
    
    # Run Flask
    proc = subprocess.Popen(
        f'"{python}" app.py',
        shell=True,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        while proc.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        proc.terminate()
        print_success("Stopped")


if __name__ == "__main__":
    main()