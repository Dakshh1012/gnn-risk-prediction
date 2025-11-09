#!/usr/bin/env python3
"""
Supply Chain Risk Intelligence Dashboard Launcher
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for the Streamlit dashboard"""
    packages = [
        "streamlit==1.25.0",
        "plotly==5.15.0", 
        "pillow==10.0.0"
    ]
    
    print("Installing required packages for Streamlit app...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n🚀 Starting Supply Chain Risk Intelligence Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🔗 Access URL: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")

if __name__ == "__main__":
    print("🚛 Supply Chain Risk Intelligence Dashboard")
    print("=" * 50)
    
    # Check if streamlit_app.py exists
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py not found in current directory")
        sys.exit(1)
    
    # Install requirements
    if install_requirements():
        print("\n✅ All packages installed successfully!")
        run_dashboard()
    else:
        print("\n❌ Failed to install required packages")
        sys.exit(1)