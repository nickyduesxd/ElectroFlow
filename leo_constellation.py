import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from datetime import datetime, timedelta
import networkx as nx
from typing import List, Tuple, Dict
import requests
import warnings
warnings.filterwarnings('ignore')

class LEOConstellation:
    """
    Manages a LEO satellite constellation with orbital mechanics and networking.
    """
    
    def __init__(self, tle_file: str = None, max_satellites: int = 100):
        """
        Purpoose for initialization constellation.
        Args:
            tle_file: Path to TLE file, or None to download Starlink data
            max_satellites: Maximum number of satellites to load - for testing
        """
        self.ts = load.timescale()
        self.satellites = []
        self.satellite_names = []
        self.max_satellites = max_satellites
        
        if tle_file is None:
            self._download_starlink_tles()
        else:
            self._load_tle_file(tle_file)
        
        print(f"Loaded {len(self.satellites)} satellites")
    
    def _download_starlink_tles(self):
        """Download recent Starlink TLE data from Celestrak."""
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            
            # Parse TLE format - 3 lines per satellite
            for i in range(0, min(len(lines), self.max_satellites * 3), 3):
                if i + 2 < len(lines):
                    name = lines[i].strip()
                    line1 = lines[i + 1].strip()
                    line2 = lines[i + 2].strip()
                    
                    sat = EarthSatellite(line1, line2, name, self.ts)
                    self.satellites.append(sat)
                    self.satellite_names.append(name)
            
            print(f"Downloaded {len(self.satellites)} Starlink satellites")
            
        except Exception as e:
            print(f"Could not download TLE data: {e}")
            self._create_synthetic_constellation()
    
    def _create_synthetic_constellation(self):
        """Create a synthetic Walker Delta constellation for testing."""
        # Walker Delta pattern: 24 satellites in 3 planes
        # Altitude: 550 km (Starlink-like)
        
        n_planes = 3
        sats_per_plane = 8
        altitude_km = 550
        inclination_deg = 53.0
        
        print(f"Creating synthetic constellation: {n_planes} planes × {sats_per_plane} sats")
        
        for plane in range(n_planes):
            raan = plane * (360.0 / n_planes)  # Right ascension of ascending node
            
            for sat in range(sats_per_plane):
                mean_anomaly = sat * (360.0 / sats_per_plane)
                
                name = f"SYNTHETIC-{plane}-{sat}"
                line1 = f"1 99999U 24001A   24001.50000000  .00000000  00000-0  00000-0 0    00"
                line2 = f"2 99999  {inclination_deg:7.4f} {raan:8.4f} 0001000 000.0000 {mean_anomaly:8.4f} 15.19000000    10"
                
                sat_obj = EarthSatellite(line1, line2, name, self.ts)
                self.satellites.append(sat_obj)
                self.satellite_names.append(name)
    
    def propagate_constellation(self, time: datetime) -> np.ndarray:
        """
        Propagate all satellites to a specific time.
        Args:
            time: Datetime to propagate to    
        Returns:
            positions: Nx3 array of ECI positions [km]
        """
        t = self.ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)
        positions = []
        for sat in self.satellites:
            geocentric = sat.at(t)
            pos = geocentric.position.km
            positions.append(pos)
        return np.array(positions)
    
    def calculate_inter_satellite_distances(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distances between all satellites.
        Args:
            positions: Nx3 array of satellite positions [km]
        Returns:
            distances: NxN array of distances [km]
        """
        n = len(positions)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances
    
    def calculate_fso_link_budget(self, distance_km: float, wavelength_nm: float = 1550, tx_power_w: float = 1.0, tx_aperture_m: float = 0.1, rx_aperture_m: float = 0.1) -> Dict:
        """
        Calculate FSO link budget for satellite-to-satellite link.
        Args:
            distance_km: Link distance [km]
            wavelength_nm: Optical wavelength [nm]
            tx_power_w: Transmitter power [W]
            tx_aperture_m: Transmitter aperture diameter [m]
            rx_aperture_m: Receiver aperture diameter [m]
        Returns:
            link_params: Dictionary with link parameters
        """
        # Convert units
        distance_m = distance_km * 1000
        wavelength_m = wavelength_nm * 1e-9
        
        # Transmitter gain (dB)
        tx_gain_db = 10 * np.log10((np.pi * tx_aperture_m / wavelength_m) ** 2)
        
        # Receiver gain (dB)
        rx_gain_db = 10 * np.log10((np.pi * rx_aperture_m / wavelength_m) ** 2)
        
        # Free space path loss (dB)
        fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(wavelength_m) - 147.55
        
        # Transmit power (dBm)
        tx_power_dbm = 10 * np.log10(tx_power_w * 1000)
        
        # Received power (dBm)
        rx_power_dbm = tx_power_dbm + tx_gain_db + rx_gain_db - fspl_db
        
        # Atmospheric loss (minimal in space)
        atmospheric_loss_db = 0.5  # Small pointing/coupling losses
        
        # Final received power
        rx_power_final_dbm = rx_power_dbm - atmospheric_loss_db
        
        # Assume Shannon capacity with optical SNR
        rx_power_w = 10 ** ((rx_power_final_dbm - 30) / 10)
        
        if rx_power_w > 1e-9:  # Viable link
            # Simple capacity model
            capacity_gbps = min(10.0, rx_power_w * 1e10)  # Cap at 10 Gbps
        else:
            capacity_gbps = 0.0
        
        return {
            'distance_km': distance_km,
            'tx_power_dbm': tx_power_dbm,
            'rx_power_dbm': rx_power_final_dbm,
            'path_loss_db': fspl_db,
            'capacity_gbps': capacity_gbps,
            'viable': capacity_gbps > 0.1  # Minimum 100 Mbps
        }
    
    def build_network_topology(self, time: datetime, max_distance_km: float = 5000, min_elevation_deg: float = 25.0) -> nx.DiGraph:
        """
        Build network topology at a specific time.
        Args:
            time: Time to build topology
            max_distance_km: Maximum ISL range [km]
            min_elevation_deg: Minimum elevation angle [deg]
        Returns:
            G: NetworkX directed graph with link attributes
        """
        # Propagate constellation
        positions = self.propagate_constellation(time)
        
        # Calculate distances
        distances = self.calculate_inter_satellite_distances(positions)
        
        # Build graph
        G = nx.DiGraph()
        
        # Add all nodes
        for i, name in enumerate(self.satellite_names):
            G.add_node(i, name=name, position=positions[i])
        
        # Add edges based on visibility and distance
        n = len(self.satellites)
        for i in range(n):
            for j in range(i + 1, n):
                dist = distances[i, j]
                if dist < max_distance_km:
                    # Calculate link budget
                    link = self.calculate_fso_link_budget(dist)
                    if link['viable']:
                        # Latency (speed of light)
                        latency_ms = dist / 299.792  # c = 299,792 km/s
                        # Add bidirectional edges
                        G.add_edge(i, j, distance=dist, latency=latency_ms, capacity=link['capacity_gbps'], rx_power=link['rx_power_dbm']) 
                        G.add_edge(j, i, distance=dist, latency=latency_ms, capacity=link['capacity_gbps'], rx_power=link['rx_power_dbm'])
        print(f"Built topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def get_network_statistics(self, G: nx.DiGraph) -> Dict:
        """Calculate network statistics."""
        if G.number_of_nodes() == 0:
            return {}   
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'density': nx.density(G),
        }
        
        if G.number_of_edges() > 0:
            latencies = [G[u][v]['latency'] for u, v in G.edges()]
            capacities = [G[u][v]['capacity'] for u, v in G.edges()]    
            stats['avg_latency_ms'] = np.mean(latencies)
            stats['avg_capacity_gbps'] = np.mean(capacities)
            stats['total_capacity_tbps'] = sum(capacities) / 1000
        return stats


def demo_constellation():
    """Demonstrate constellation simulation."""
    # Create constellation
    constellation = LEOConstellation(max_satellites=50)
    
    # Simulate at current time
    current_time = datetime.utcnow()
    print(f"\nSimulating at: {current_time}")
    
    # Build network topology
    print("\nBuilding network topology...")
    G = constellation.build_network_topology(current_time, max_distance_km=5000)
    
    # Print statistics
    stats = constellation.get_network_statistics(G)
    for key, value in stats.items():
        print(f"{key:25s}: {value:.3f}")
    
    return constellation, G

if __name__ == "__main__":
    constellation, G = demo_constellation()
