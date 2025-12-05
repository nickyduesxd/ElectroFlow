#!/bin/bash

# LEO Satellite Electrical Flow Routing - Dependency Setup
# Run this first to install all required packages

echo "========================================="
echo "LEO Satellite Routing - Setup"
echo "========================================="
echo ""

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

echo ""
echo "✓ All dependencies installed!"
echo ""
echo "========================================="
echo "Next Steps:"
echo "========================================="
echo "1. Run: python leo_constellation.py"
echo "   (Downloads TLE data, tests orbital mechanics)"
echo ""
echo "2. Run: python leo_electrical_routing.py"
echo "   (Implements electrical flow routing)"
echo ""
echo "3. Start experimenting with parameters!"
echo "========================================="
