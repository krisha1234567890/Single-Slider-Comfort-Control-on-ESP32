# HIL Comfort Control System

Hardware-in-the-Loop HVAC comfort control using ESP32 and Python.

## Features
- Single-slider Comfort Factor (0-100) control
- 3D Hilbert curve dimensionality reduction
- 2-16-3 Neural Network (99 parameters)
- Bidirectional PID control (±100 output)
- 30-run statistical analysis
- Real-time plotting GUI

## Hardware Requirements
- ESP32-WROOM-32
- USB cable for serial communication

## Software Requirements
- Python 3.12+
- Arduino IDE (for ESP32)

## Installation
```bash
pip install pyserial numpy matplotlib tkinter
