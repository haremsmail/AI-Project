🔧 COMPLETE SETUP INSTRUCTIONS
================================

The PSO Visualization project requires minimal setup.
Follow these steps to get started.


1️⃣ VERIFY PYTHON VERSION
  ┌──────────────────────────────────────────────┐
  │ Open terminal/command prompt and run:        │
  │   python --version                           │
  │                                              │
  │ Expected output: Python 3.7.x or higher     │
  └──────────────────────────────────────────────┘


2️⃣ VERIFY TKINTER INSTALLATION
  ┌──────────────────────────────────────────────┐
  │ Run:                                          │
  │   python -c "import tkinter; print('OK')"   │
  │                                              │
  │ Expected output: OK                          │
  │                                              │
  │ If error, install Tkinter (see below)        │
  └──────────────────────────────────────────────┘


3️⃣ IF TKINTER NOT INSTALLED
  
  Windows:
    → Usually included by default
    → Reinstall Python and check 'tcl/tk and IDLE' option
  
  macOS:
    → Install Python from python.org (includes Tkinter)
    → OR: brew install python-tk
  
  Ubuntu/Debian Linux:
    → sudo apt-get update
    → sudo apt-get install python3-tk
  
  Fedora/RHEL:
    → sudo dnf install python3-tkinter
  
  Arch Linux:
    → sudo pacman -S tk


4️⃣ DOWNLOAD PROJECT FILES
  
  Ensure you have these files in week4 folder:
    ✓ main.py
    ✓ ui.py
    ✓ pso_logic.py
    ✓ requirements.txt
    ✓ README.md


5️⃣ RUN THE APPLICATION
  
  ┌──────────────────────────────────────────────┐
  │ Navigate to week4 folder:                    │
  │   cd path/to/week4                           │
  │                                              │
  │ Run:                                         │
  │   python main.py                            │
  │                                              │
  │ Window should open in 2-3 seconds!          │
  └──────────────────────────────────────────────┘


6️⃣ VERIFY INSTALLATION
  
  The window should show:
    • Canvas on left (650x650 dark background)
    • Control panel on right
    • Parameter input fields
    • Button bar (▶ Start, ⏸ Pause, 🔄 Reset)
    • History list
    • Speed slider


7️⃣ TEST RUN
  
  1. Accept default parameters
  2. Click ▶ Start
  3. You should see:
     • Particles appear in the canvas
     • They start moving toward center
     • History list updates with each iteration
     • Info label shows iteration count and fitness


⚙️ TROUBLESHOOTING

Q: "ModuleNotFoundError: No module named 'tkinter'"
A: Install Tkinter (see step 3 above)

Q: "Python not found"
A: Add Python to your PATH or use full path: /usr/bin/python3 main.py

Q: Window appears but particles don't move
A: Click ▶ Start button, make sure Max Iterations > 1

Q: Window is very small or cut off
A: Check your display scaling. Window is 1100x750 pixels

Q: Nothing happens when I click buttons
A: Make sure parameters are valid numbers (no letters)

Q: History list is empty
A: You need to click ▶ Start first


📚 FILE DESCRIPTIONS

main.py (15 lines)
  - Entry point for application
  - Creates Tkinter root window
  - Initializes PSOApp
  
pso_logic.py (150 lines)
  - Particle class: Models one swarm member
  - PSO algorithm implementation
  - Completely manual (no libraries)
  
ui.py (450 lines)
  - PSOApp class: Main GUI application
  - Canvas rendering
  - Event handlers
  - Animation loop

requirements.txt
  - Documentation only (no packages needed)

README.md
  - Complete documentation

QUICKSTART.txt
  - Quick reference guide


🚀 FIRST TIME USERS

Recommended first run:
  • Keep all default parameters
  • Set Max Iterations to 50
  • Set Animation Speed to 50ms
  • Click Start and observe

Then try:
  • Change Number of Particles to 20
  • Change Goal to different position
  • Adjust w, c1, c2 values
  • Try different Max Iterations


💡 PRO TIPS

• Reset between runs to clear old data
• Use Pause button to study the canvas
• Select iterations in history to replay them
• Slow animation speed to see particle details
• Open in full screen for better viewing


✅ SUCCESSFUL SETUP CHECK

You're ready if:
  ✓ Python 3.7+ installed
  ✓ Tkinter imports without error
  ✓ All 3 .py files present
  ✓ main.py runs and window appears
  ✓ Buttons and sliders work
  ✓ Animation starts and history updates


📞 IF PROBLEMS PERSIST

1. Check that all .py files are in same folder
2. Verify Python version (python --version)
3. Test Tkinter separately (see step 2)
4. Try full Python path: /usr/bin/python3 main.py
5. On Windows, try: python -m main
6. Check for permission issues (chmod +x main.py on Linux/Mac)


🎓 NEXT STEPS

After successful setup:
  1. Read README.md for complete guide
  2. Try QUICKSTART.txt for quick reference
  3. Experiment with different PSO parameters
  4. Study the code in pso_logic.py to learn PSO
  5. Modify ui.py to customize the visualization


═══════════════════════════════════════════════════════════════════

Setup complete! Enjoy using PSO Visualization! 🚀

For questions, see README.md or check code comments.
