#!/bin/bash

# LEO Satellite Electrical Flow Routing - Dependency Setup
# Run this first to install all required packages

echo "Installing core dependencies..."

# Core scientific packages
pip install numpy scipy matplotlib --break-system-packages -q

# Satellite orbital mechanics
pip install skyfield --break-system-packages -q

# Network analysis
pip install networkx --break-system-packages -q

# Data handling
pip install pandas --break-system-packages -q

# Optional: 3D visualization
pip install plotly --break-system-packages -q 2>/dev/null || echo "Plotly optional, continuing..."

