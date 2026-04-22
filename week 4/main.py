"""
main.py - Entry point for PSO Visualization App

Run this file to start the application:
    python main.py
"""

import tkinter as tk
from ui import PSOApp

if __name__ == "__main__":
    root = tk.Tk()
    app = PSOApp(root)
    root.mainloop()
