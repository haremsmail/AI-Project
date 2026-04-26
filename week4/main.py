"""
═══════════════════════════════════════════════════════════════════
  PSO VISUALIZATION - PARTICLE SWARM OPTIMIZATION
  
  A complete implementation of the Particle Swarm Optimization 
  algorithm with interactive 2D visualization using Tkinter.
  
  NO EXTERNAL LIBRARIES USED EXCEPT TKINTER (built-in)
═══════════════════════════════════════════════════════════════════

Author: haremsmail (AI-Project)
Version: 2.0
Date: 2026
License: MIT

QUICK START:
  python main.py

REQUIREMENTS:
  - Python 3.7 or higher
  - Tkinter (included with Python by default)

FEATURES:
  ✓ Manual PSO implementation (no libraries)
  ✓ Real-time particle swarm visualization
  ✓ Configurable PSO parameters
  ✓ Velocity vectors and particle trails
  ✓ Generation history with replay
  ✓ Start/Pause/Reset controls
  ✓ Adjustable animation speed
  ✓ Professional dark theme UI

═══════════════════════════════════════════════════════════════════
"""
"this is the library of python "
import tkinter as tk
from ui import PSOApp


def main():
    """Initialize and run the PSO Visualization application."""
    root = tk.Tk()
    app = PSOApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
