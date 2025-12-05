# Electrical Flow Routing for LEO Satellite Constellations

## Research Overview

This project implements **electrical flow routing** for LEO satellite networks, inspired by how electrons naturally distribute current through electrical circuits according to Ohm's law and Kirchhoff's laws.

### Core Hypothesis
> Electrons naturally load-balance across parallel paths based on resistance. Can we leverage this physics-inspired approach for optimal traffic routing in satellite mega-constellations?

---

## Algorithm Explained

### The Circuit Analogy

| Network Concept | Electrical Analog |
|----------------|------------------|
| Satellite node | Electrical junction |
| Inter-satellite link (ISL) | Resistor |
| Data traffic | Electric current |
| Source satellite | Positive voltage terminal |
| Destination satellite | Ground (0V) |
| Routing decision | Natural current distribution |

### Mathematical Foundation

**1. Resistance Mapping**
```
R(link) = α·(latency) + β·(1/capacity)
```
- Higher latency → Higher resistance (avoid slow links)
- Higher capacity → Lower resistance (prefer high bandwidth links)
- Parameters α and β control the trade-off

**2. Kirchhoff's Current Law (KCL)**
At every node except source/destination:
```
∑ I_in = ∑ I_out
```
Current flowing in must equal current flowing out (conservation of flow).

**3. Ohm's Law**
For each link:
```
V_u - V_v = I_uv · R_uv
```
Voltage difference drives current through resistance.

**4. System of Equations**
Combining these gives the **Laplacian system**:
```
L · V = b
```
where:
- **L** = Graph Laplacian matrix (conductances)
- **V** = Node voltage vector (unknowns)
- **b** = Current injection vector (+1 at source, -1 at dest)

**5. Solution**
Solve for voltages V, then compute flows on each edge:
```
Flow(u→v) = Conductance(u,v) · (V_u - V_v)
```

### Why This Works

1. **Natural Load Balancing**: Current automatically splits across parallel paths proportional to their conductances (inverse resistances)
2. **Optimal Distribution**: Follows principle of least energy dissipation
3. **Fast Computation**: Sparse linear system, O(n log n) with advanced solvers
4. **Physical Realism**: Based on proven physical laws, not heuristics

---

## 🔧 Implementation Details

### Key Classes

**LEOConstellation** (`leo_constellation.py`)
- Downloads real TLE data (Starlink)
- SGP4 orbital propagation
- FSO link budget calculations
- Builds time-varying network topology

**ElectricalFlowRouter** (`leo_electrical_routing.py`)
- Maps network to electrical circuit
- Solves Laplacian system for voltages
- Computes edge flows using Ohm's law
- Extracts routing paths
- Compares with Dijkstra

### Critical Parameters

```python
# In _calculate_resistance()
alpha = 1.0  # Weight for latency
beta = 1.0   # Weight for capacity

# Resistance formula
R = α * (latency/100) + β * (10/capacity)
```

**Tuning Guide:**
- `α > β`: Prioritize low latency (minimize delay)
- `β > α`: Prioritize high capacity (maximize throughput)
- `α = β`: Balanced optimization

### Network Topology

```python
# In build_network_topology()
max_distance_km = 5000  # Maximum ISL range
```

**LEO Typical Ranges:**
- Intra-plane: 500-2000 km
- Inter-plane: 2000-5000 km
- Cross-seam: 3000-6000 km

---

## Running the Code

### Setup
```bash
chmod +x setup_leo.sh
./setup_leo.sh
```

### Run Constellation Simulator
```bash
python leo_constellation.py
```

This will:
1. Download Starlink TLE data
2. Propagate orbits to current time
3. Calculate FSO link budgets
4. Build network topology
5. Print statistics

### Run Electrical Routing
```bash
python leo_electrical_routing.py
```

This will:
1. Load constellation
2. Build topology
3. Run electrical flow routing
4. Run Dijkstra for comparison
5. Print detailed comparison

---

### Key Insights

**Advantages:**
- Discovers multiple parallel paths automatically
- Natural load balancing across paths
- Scales well to large constellations
- Considers both latency and capacity

**Trade-offs:**
- Slightly higher latency than pure shortest path
- More computation than simple Dijkstra
- May use lower-capacity backup paths

---

## Debugging

### Common Issues

**1. No viable links in constellation**
```
Error: No viable links in constellation
```
**Fix:** Increase `max_distance_km` or reduce `min_elevation_deg`

**2. Singular matrix in solver**
```
Warning: Could not solve electrical system
```
**Fix:** Check for disconnected nodes, increase minimum resistance

**3. Numerical instability**
```
Resistance values too small/large
```
**Fix:** Normalize latency and capacity in resistance calculation

### Performance Optimization

**For large constellations (500+ satellites):**

```python
# Use iterative solver instead of direct
from scipy.sparse.linalg import cg  # Conjugate gradient

# Replace spsolve with:
V_reduced, info = cg(L_reduced, b_reduced, tol=1e-6)
```

**Memory optimization:**
```python
# Limit edges based on capacity threshold
if link['capacity_gbps'] < 1.0:
    continue  # Skip low-capacity links
```

---

*Last updated: December 2025*
