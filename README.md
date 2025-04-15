# MusicDepth
This repository showcases a series of experiments in sketch-based interaction, where hand-drawn shapes on paper are transformed into interactive digital controls using computer vision. The project explores how simple drawings — when tracked in real-time — can become functional input devices, music controllers, and creative instruments.

🧩 Project Components
🖱️ 1. Paper Mouse
Turn a piece of paper into a functioning mouse by drawing a control shape (e.g. triangle) and using computer vision to track its movement. Supports motion tracking, click emulation, and cursor control.

🎛️ 2. Drawn Music Player Controls
Draw your own play, pause, next, and volume buttons on paper. The system recognizes these hand-drawn UI elements and maps them to real music control actions. Interact with the paper like a touch interface.

🎶 3. Sketch-to-Sound (Drawn GarageBand)
Draw shapes like drums, guitars, or trumpets on paper. When tapped, these shapes trigger corresponding instrument sounds — turning your desk into a live sketch-based soundboard.


## Install

brew install librealsense
brew install opencv
brew install glfw

#then install requirements file
pip3 install -r requirements.txt

##Virtual Env
python3 -m venv myenv
source myenv/bin/activate

#run main.py
python3 main.py

#run depth.py
sudo python3 depth.py
