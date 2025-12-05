import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from leo_constellation import LEOConstellation
from datetime import datetime
import os


class ElectricalFlowRouter:
    """
    Routes traffic using electrical flow principles.
    The network is modeled as an electrical circuit where:
    - Nodes = Electrical junctions
    - Edges = Resistors
    - Source node = Positive voltage
    - Destination node = Ground (0V)
    - Flow = Current following Ohm's law (V = IR)
    """
    
    def __init__(self, G: nx.DiGraph, alpha: float = 2.0, beta: float = 0.5):
        """
        Initialize router with network topology.
        Args:
            G: NetworkX directed graph with 'latency' and 'capacity' attributes
            alpha: Resistance weight for latency (default: 2.0 = prioritize latency)
            beta: Resistance weight for capacity (default: 0.5 = secondary consideration)
        """
        self.G = G
        
        # Only use nodes that are actually in the graph's edges
        nodes_in_edges = set()
        for u, v in G.edges():
            nodes_in_edges.add(u)
            nodes_in_edges.add(v)
        
        self.node_list = sorted(list(nodes_in_edges))
        self.n_nodes = len(self.node_list)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.alpha = alpha
        self.beta = beta
        
        print(f"Initialized electrical flow router: {self.n_nodes} nodes (with edges)")
        print(f"Resistance weights: α={alpha:.1f} (latency), β={beta:.1f} (capacity)")
        
        if self.n_nodes < G.number_of_nodes():
            isolated = G.number_of_nodes() - self.n_nodes
    
    def _calculate_resistance(self, latency_ms: float, capacity_gbps: float, alpha: float = 2.0, beta: float = 0.5) -> float:
        """
        Map network parameters to electrical resistance.
        Args:
            latency_ms: Link latency [ms]
            capacity_gbps: Link capacity [Gbps]
            alpha: Weight for latency component (higher = prioritize low latency)
            beta: Weight for capacity component (higher = prioritize high capacity)
        Returns:
            resistance: Effective resistance [Ω]
        Resistance model:
            R = α * (latency) + β * (1/capacity)
            
        """
        # Normalize to prevent numerical issues
        latency_normalized = latency_ms / 100.0  # ~100ms typical ISL
        capacity_normalized = 10.0 / max(capacity_gbps, 0.1)  # 10 Gbps typical
        resistance = alpha * latency_normalized + beta * capacity_normalized
        # Minimum resistance to avoid singularities
        return max(resistance, 1e-6)
    
    def _build_laplacian_matrix(self, source: int, dest: int) -> Tuple[csr_matrix, np.ndarray]:
        """
        Build graph Laplacian matrix for electrical flow computation.
        The Laplacian L is defined as:
            L = D - A: where D = Degree matrix (diagonal) and A = Weighted adjacency matrix (conductances)
        For electrical flow:
            L @ V = I: where V = Node voltages and I = Current injection vector
        Args:
            source: Source node
            dest: Destination node  
        Returns:
            L: Laplacian matrix (sparse)
            b: Right-hand side vector (current injection)
        """
        n = self.n_nodes
        L = lil_matrix((n, n))
        source_idx = self.node_to_idx[source]
        dest_idx = self.node_to_idx[dest]
        
        # Build Laplacian from conductances (1/resistance)
        edge_count = 0
        for u, v, data in self.G.edges(data=True):
            resistance = self._calculate_resistance(
                data['latency'], 
                data['capacity'],
                self.alpha,
                self.beta
            )
            conductance = 1.0 / resistance
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            # L[i,i] = sum of conductances at node i
            L[u_idx, u_idx] += conductance
            L[v_idx, v_idx] += conductance
            # L[i,j] = -conductance between i and j
            L[u_idx, v_idx] -= conductance
            L[v_idx, u_idx] -= conductance
            edge_count += 1
        
        if edge_count == 0:
            return None, None
        
        # Current injection vector: +1 at source, -1 at destination
        b = np.zeros(n)
        b[source_idx] = 1.0
        b[dest_idx] = -1.0
        
        # Convert to CSR format for efficient solving
        L = L.tocsr()
        
        return L, b
    
    def _calculate_load_balance_metrics(self, edge_flows: Dict, paths: List[List[int]]) -> Dict:
        """
        Calculate load balancing metrics for the flow distribution.
        Metrics:
        - Flow entropy: Measures how evenly flow is distributed (higher = better)
        - Max/avg flow ratio: Ratio of maximum to average flow (lower = better)
        - Effective paths: Number of paths carrying significant flow
        - Flow concentration: Gini coefficient of flow distribution (0 = perfect balance)
        Args:
            edge_flows: Dictionary of edge flows
            paths: List of routing paths
        Returns:
            metrics: Dictionary of load balancing metrics
        """
        if not edge_flows:
            return {
                'entropy': 0.0,
                'max_avg_ratio': 0.0,
                'effective_paths': 0,
                'gini_coefficient': 0.0,
                'flow_distribution': 'N/A'
            }
        
        # Get absolute flows and filter near-zero values
        flows = np.array([abs(f) for f in edge_flows.values()])
        flows = flows[flows > 1e-9]
        
        if len(flows) == 0:
            return {
                'entropy': 0.0,
                'max_avg_ratio': 0.0,
                'effective_paths': 0,
                'gini_coefficient': 0.0,
                'flow_distribution': 'No flow'
            }
        
        # Normalize flows to probabilities
        total_flow = np.sum(flows)
        probabilities = flows / total_flow
        
        # Shannon Entropy (higher = more distributed): Max entropy = log(n) where n = number of edges
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        max_entropy = np.log2(len(flows))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Max/Average Flow Ratio (lower = more balanced)
        max_flow = np.max(flows)
        avg_flow = np.mean(flows)
        max_avg_ratio = max_flow / avg_flow if avg_flow > 0 else 0
        
        # Effective number of paths (using Herfindahl index): Higher = more paths carrying significant flow
        herfindahl = np.sum(probabilities ** 2)
        effective_paths = 1.0 / herfindahl if herfindahl > 0 else 0
        
        # Gini Coefficient (0 = perfect equality, 1 = perfect inequality)
        sorted_flows = np.sort(flows)
        n = len(sorted_flows)
        cumsum = np.cumsum(sorted_flows)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_flows))) / (n * np.sum(sorted_flows)) - (n + 1) / n
        
        # Flow distribution characterization
        if gini < 0.3:
            distribution = "Excellent - Highly balanced"
        elif gini < 0.5:
            distribution = "Good - Well distributed"
        elif gini < 0.7:
            distribution = "Fair - Moderately concentrated"
        else:
            distribution = "Poor - Highly concentrated"
        
        return {
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'max_avg_ratio': max_avg_ratio,
            'effective_paths': effective_paths,
            'gini_coefficient': gini,
            'flow_distribution': distribution,
            'active_edges': len(flows),
            'total_edges': len(edge_flows)
        }
    
    def _trace_flow_path(self, source: int, dest: int, edge_flows: Dict, max_flow: float) -> List[List[int]]:
        """
        Manually trace flow from source to destination using greedy approach.
        Args:
            source: Source node
            dest: Destination node
            edge_flows: Dictionary of edge flows
            max_flow: Maximum flow value for normalization
        Returns:
            paths: List containing traced path(s)
        """
        path = [source]
        current = source
        visited = {source}
        
        # Greedy trace: follow highest outgoing flow
        max_iterations = 20
        for _ in range(max_iterations):
            if current == dest:
                return [path]
            
            # Find best next hop - highest positive flow
            best_next = None
            best_flow = -float('inf')
            
            for (u, v), flow in edge_flows.items():
                if u == current and v not in visited and flow > 0:
                    if flow > best_flow:
                        best_flow = flow
                        best_next = v
            
            if best_next is None:
                # Dead end - return empty
                return []
            
            path.append(best_next)
            visited.add(best_next)
            current = best_next
        
        return []
    
    def compute_electrical_flow(self, source: int, dest: int, demand_gbps: float = 1.0) -> Dict:
        """
        Compute electrical flow from source to destination.
        Solves Kirchhoff's equations:
            L @ V = I
        Then computes edge flows using Ohm's law:
            Flow(u,v) = Conductance(u,v) * (V[u] - V[v])
        Args:
            source: Source node
            dest: Destination node  
            demand_gbps: Traffic demand [Gbps]
        Returns:
            result: Dictionary with voltages, flows, and paths
        """
        start_time = time.time()
        
        # Verify source and dest are in routing graph
        if source not in self.node_to_idx or dest not in self.node_to_idx:
            print(f"Error: Source ({source}) or dest ({dest}) not in routing graph")
            print(f"Routing graph has nodes: {min(self.node_to_idx.keys())} to {max(self.node_to_idx.keys())}")
            print(f"Total routing nodes: {len(self.node_to_idx)}")
            
            # Check if they exist in original graph
            if source in self.G.nodes() and dest in self.G.nodes():
                print(f"Nodes exist in original graph but are isolated")
            
            return None
        
        # Build system of equations
        L, b = self._build_laplacian_matrix(source, dest)
        
        if L is None or b is None:
            print(f"Failed to build Laplacian matrix")
            return None
        
        # Ground the destination (V[dest] = 0) which is done by removing the destination equation
        dest_idx = self.node_to_idx[dest]
        
        # Create reduced system (all nodes except destination)
        indices = [i for i in range(self.n_nodes) if i != dest_idx]
        L_reduced = L[indices, :][:, indices]
        b_reduced = b[indices]
        
        # Check matrix condition
        if L_reduced.shape[0] == 0:
            print(f"Error: Empty reduced system")
            return None
        
        # Solve for voltages
        try:
            V_reduced = spsolve(L_reduced, b_reduced)
            
            # Check for NaN or Inf
            if np.any(np.isnan(V_reduced)) or np.any(np.isinf(V_reduced)):
                print(f"Solver returned invalid values (NaN/Inf) - the network has disconnected components")
                return None
            
            # Reconstruct full voltage vector
            V = np.zeros(self.n_nodes)
            for i, idx in enumerate(indices):
                V[idx] = V_reduced[i]
            V[dest_idx] = 0.0  # Ground
            
        except Exception as e:
            print(f"Could not solve electrical system: {e}")
            return None
        
        # Compute edge flows using Ohm's law
        edge_flows = {}
        total_flow = 0.0
        
        for u, v, data in self.G.edges(data=True):
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            resistance = self._calculate_resistance(
                data['latency'],
                data['capacity'],
                self.alpha,
                self.beta
            )
            conductance = 1.0 / resistance
            
            # Flow = Conductance * Voltage difference
            voltage_diff = V[u_idx] - V[v_idx]
            flow = conductance * voltage_diff * demand_gbps
            edge_flows[(u, v)] = flow
            if u == source or v == dest:
                total_flow += abs(flow)
        
        compute_time = time.time() - start_time
        
        # Extract paths (trace high-flow edges)
        paths = self._extract_paths_from_flows(source, dest, edge_flows)
        
        # Debug
        if not paths:
            positive_flows = sum(1 for f in edge_flows.values() if abs(f) > 1e-9)
            print(f"No paths extracted from {positive_flows} non-zero flows")
            print(f"Max flow magnitude: {max(abs(f) for f in edge_flows.values()):.6f}")
        
        # Calculate load balancing metrics
        load_balance_metrics = self._calculate_load_balance_metrics(edge_flows, paths)
        
        result = {
            'voltages': {self.node_list[i]: V[i] for i in range(self.n_nodes)},
            'edge_flows': edge_flows,
            'paths': paths,
            'total_flow': total_flow,
            'compute_time_ms': compute_time * 1000,
            'source': source,
            'dest': dest,
            'demand_gbps': demand_gbps,
            'load_balance': load_balance_metrics
        }
        
        return result
    
    def _extract_paths_from_flows(self, source: int, dest: int, edge_flows: Dict, threshold: float = 0.001) -> List[List[int]]:
        """
        Extract routing paths from edge flows.
        Traces high-flow edges from source to destination.
        Args:
            source: Source node
            dest: Destination node
            edge_flows: Dictionary of edge flows
            threshold: Minimum flow to consider (fraction of max, default 0.001 = 0.1%)
        Returns:
            paths: List of paths (each path is list of nodes)
        """
        if not edge_flows:
            return []
        
        # Find maximum absolute flow
        max_flow = max(abs(f) for f in edge_flows.values())
        if max_flow == 0:
            return []
        
        # Build subgraph of significant flows
        thresholds = [threshold, threshold * 0.1, threshold * 0.01]
        
        for thresh in thresholds:
            G_flow = nx.DiGraph()
            for (u, v), flow in edge_flows.items():
                if abs(flow) >= thresh * max_flow:
                    G_flow.add_edge(u, v, flow=abs(flow), weight=1.0/max(abs(flow), 1e-9))
            
            # Check if source and dest are in the flow graph
            if source not in G_flow or dest not in G_flow:
                continue
            
            # Find paths
            try:
                # Find shortest path by num of hops first
                shortest_path = nx.shortest_path(G_flow, source, dest)
                paths = [shortest_path]
                
                # Find alternative paths
                try:
                    for path in nx.all_simple_paths(G_flow, source, dest, cutoff=15):
                        if path not in paths:
                            paths.append(path)
                        if len(paths) >= 5:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                
                if paths:
                    return paths
                    
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        
        # If no paths, try to trace flow from source to dest manually
        return self._trace_flow_path(source, dest, edge_flows, max_flow)
    
    def compute_ospf_routing(self, source: int, dest: int) -> Dict:
        """
        Compute OSPF (Open Shortest Path First) routing.
        OSPF uses Dijkstra's algorithm with link costs based on:
        - Reference bandwidth / link bandwidth
        - Default reference: 100 Mbps
        Args:
            source: Source node
            dest: Destination node
        Returns:
            result: Dictionary with OSPF routing information
        """
        start_time = time.time()
        
        # Calculate OSPF costs for each link: Cost = reference_bandwidth / link_bandwidth
        reference_bw_gbps = 10.0  # 10 Gbps
        
        ospf_costs = {}
        for u, v, data in self.G.edges(data=True):
            # OSPF cost based on bandwidth
            cost = reference_bw_gbps / max(data['capacity'], 0.1)
            ospf_costs[(u, v)] = cost
        
        # Set edge weights for OSPF calculation
        nx.set_edge_attributes(self.G, ospf_costs, 'ospf_cost')
        
        try:
            # Find shortest path using OSPF costs
            path = nx.dijkstra_path(self.G, source, dest, weight='ospf_cost')
            
            # Calculate path metrics
            path_latency = sum(self.G[path[i]][path[i+1]]['latency'] 
                             for i in range(len(path)-1))
            
            path_capacity = min(self.G[path[i]][path[i+1]]['capacity']
                              for i in range(len(path)-1))
            
            total_cost = sum(self.G[path[i]][path[i+1]]['ospf_cost']
                           for i in range(len(path)-1))
            
        except nx.NetworkXNoPath:
            path = []
            path_latency = float('inf')
            path_capacity = 0
            total_cost = float('inf')
        
        compute_time = (time.time() - start_time) * 1000
        
        result = {
            'path': path,
            'path_length': len(path) - 1 if path else 0,
            'latency_ms': path_latency,
            'capacity_gbps': path_capacity,
            'total_cost': total_cost,
            'compute_time_ms': compute_time,
            'load_balanced': False
        }
        
        return result
    
    def compute_ecmp_routing(self, source: int, dest: int, max_paths: int = 4) -> Dict:
        """
        Compute ECMP (Equal-Cost Multi-Path) routing.
        ECMP finds all shortest paths with equal cost and load-balances
        traffic across them using hash-based selection.
        Args:
            source: Source node
            dest: Destination node
            max_paths: Maximum number of equal-cost paths
        Returns:
            result: Dictionary with ECMP routing information
        """
        start_time = time.time()
        
        # Latency as the cost metric
        try:
            # Find shortest path length
            shortest_length = nx.dijkstra_path_length(self.G, source, dest, weight='latency') 
            # Use k-shortest paths with a VERY strict cutoff to avoid long searches
            tolerance = 1e-6
            all_paths = []
            
            # Only search with small cutoff to limit computation
            max_cutoff = 25  # Limit path length exploration
            
            try:
                # Use simple k-shortest paths with strict limits
                for path in nx.shortest_simple_paths(self.G, source, dest, weight='latency'):
                    path_length = sum(self.G[path[i]][path[i+1]]['latency']
                                    for i in range(len(path)-1))
                    
                    # Only accept paths within tolerance
                    if abs(path_length - shortest_length) <= tolerance:
                        all_paths.append(path)
                        if len(all_paths) >= max_paths:
                            break
                    elif path_length > shortest_length + tolerance:
                        # Paths are ordered by cost, so we can stop
                        break
                    
                    # Stop if path is too long
                    if len(path) > max_cutoff:
                        break 
            except (StopIteration, nx.NetworkXNoPath):
                pass
            
            if not all_paths:
                # Fall back to single shortest path
                all_paths = [nx.dijkstra_path(self.G, source, dest, weight='latency')]
            
            # Calculate aggregate metrics
            avg_latency = np.mean([sum(self.G[p[i]][p[i+1]]['latency']
                                      for i in range(len(p)-1))
                                  for p in all_paths])
            
            avg_capacity = np.mean([min(self.G[p[i]][p[i+1]]['capacity']
                                       for i in range(len(p)-1))
                                   for p in all_paths])
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            all_paths = []
            avg_latency = float('inf')
            avg_capacity = 0
        
        compute_time = (time.time() - start_time) * 1000
        
        result = {
            'paths': all_paths,
            'num_paths': len(all_paths),
            'avg_latency_ms': avg_latency,
            'avg_capacity_gbps': avg_capacity,
            'compute_time_ms': compute_time,
            'load_balanced': len(all_paths) > 1
        }
        
        return result
    
    def compare_all_algorithms(self, source: int, dest: int) -> Dict:
        """
        Compare all routing algorithms: Electrical Flow, Dijkstra, OSPF, and ECMP.
        Args:
            source: Source node
            dest: Destination node
        Returns:
            comparison: Dictionary with metrics for all algorithms
        """
        print(f"ROUTING FROM {source} -> {dest}")
        
        # Electrical Flow
        print("Computing Electrical Flow...")
        elec_result = self.compute_electrical_flow(source, dest)
        
        if elec_result is None:
            print("Electrical flow failed")
            return None
        
        if elec_result['paths']:
            elec_path = elec_result['paths'][0]
            elec_latency = sum(self.G[elec_path[i]][elec_path[i+1]]['latency']
                             for i in range(len(elec_path)-1))
            elec_capacity = min(self.G[elec_path[i]][elec_path[i+1]]['capacity']
                              for i in range(len(elec_path)-1))
        else:
            elec_path = []
            elec_latency = float('inf')
            elec_capacity = 0
        
        print(f"Found {len(elec_result['paths'])} paths in {elec_result['compute_time_ms']:.3f} ms")
        
        # Dijkstra
        print("Computing Dijkstra Shortest Path...")
        start_time = time.time()
        try:
            dijkstra_path = nx.dijkstra_path(self.G, source, dest, weight='latency')
            dijkstra_time = (time.time() - start_time) * 1000  
            dijkstra_latency = sum(self.G[dijkstra_path[i]][dijkstra_path[i+1]]['latency']
                                  for i in range(len(dijkstra_path)-1))
            dijkstra_capacity = min(self.G[dijkstra_path[i]][dijkstra_path[i+1]]['capacity']
                                   for i in range(len(dijkstra_path)-1))
        except nx.NetworkXNoPath:
            dijkstra_path = []
            dijkstra_time = 0
            dijkstra_latency = float('inf')
            dijkstra_capacity = 0
        
        print(f"Found path in {dijkstra_time:.3f} ms")
        
        # OSPF
        print("Computing OSPF Routing...")
        ospf_result = self.compute_ospf_routing(source, dest)
        print(f"Found path in {ospf_result['compute_time_ms']:.3f} ms")
        
        # ECMP
        print("Computing ECMP Multi-Path...")
        ecmp_result = self.compute_ecmp_routing(source, dest)
        print(f"Found {ecmp_result['num_paths']} paths in {ecmp_result['compute_time_ms']:.3f} ms")
        
        # Compile results
        comparison = {
            'electrical': {
                'paths': elec_result['paths'],
                'num_paths': len(elec_result['paths']),
                'primary_latency_ms': elec_latency,
                'primary_capacity_gbps': elec_capacity,
                'compute_time_ms': elec_result['compute_time_ms'],
                'load_balanced': len(elec_result['paths']) > 1,
                'load_balance_metrics': elec_result['load_balance']
            },
            'dijkstra': {
                'path': dijkstra_path,
                'latency_ms': dijkstra_latency,
                'capacity_gbps': dijkstra_capacity,
                'compute_time_ms': dijkstra_time,
                'load_balanced': False
            },
            'ospf': ospf_result,
            'ecmp': ecmp_result,
            'baseline_latency': dijkstra_latency,  # Use Dijkstra as baseline
            'edge_flows': elec_result.get('edge_flows', {})  # Add for visualization
        }
        return comparison
    
    def print_comprehensive_comparison(self, comparison: Dict):
        """Pretty print comprehensive routing algorithm comparison."""
        print(" "*20 + "ROUTING ALGORITHM COMPARISON")
        
        baseline = comparison['baseline_latency']
        
        # Header
        print(f"\n{'Algorithm':<20} {'Paths':<8} {'Latency':<12} {'Capacity':<12} {'Time':<12} {'LB':<6}")
        print("-" * 80)
        
        # Electrical Flow
        elec = comparison['electrical']
        print(f"{'Electrical':<20} {elec['num_paths']:<8} "
              f"{elec['primary_latency_ms']:>8.2f} ms  "
              f"{elec['primary_capacity_gbps']:>8.2f} Gbps "
              f"{elec['compute_time_ms']:>8.3f} ms  "
              f"{'Yes' if elec['load_balanced'] else 'No':<6}")
        
        # Dijkstra
        dijk = comparison['dijkstra']
        print(f"{'Dijkstra':<20} {1:<8} "
              f"{dijk['latency_ms']:>8.2f} ms  "
              f"{dijk['capacity_gbps']:>8.2f} Gbps "
              f"{dijk['compute_time_ms']:>8.3f} ms  "
              f"{'No':<6}")
        
        # OSPF
        ospf = comparison['ospf']
        print(f"{'OSPF':<20} {1:<8} "
              f"{ospf['latency_ms']:>8.2f} ms  "
              f"{ospf['capacity_gbps']:>8.2f} Gbps "
              f"{ospf['compute_time_ms']:>8.3f} ms  "
              f"{'No':<6}")
        
        # ECMP
        ecmp = comparison['ecmp']
        print(f"{'ECMP':<20} {ecmp['num_paths']:<8} "
              f"{ecmp['avg_latency_ms']:>8.2f} ms  "
              f"{ecmp['avg_capacity_gbps']:>8.2f} Gbps "
              f"{ecmp['compute_time_ms']:>8.3f} ms  "
              f"{'Yes' if ecmp['load_balanced'] else 'No':<6}")
        
        # Performance analysis
        print("PERFORMANCE ANALYSIS")
        
        print("\nLatency Comparison (relative to Dijkstra baseline):")
        if baseline > 0 and baseline < float('inf'):
            print(f"   Electrical:  {(elec['primary_latency_ms']/baseline - 1)*100:+.1f}%")
            print(f"   OSPF:        {(ospf['latency_ms']/baseline - 1)*100:+.1f}%")
            print(f"   ECMP:        {(ecmp['avg_latency_ms']/baseline - 1)*100:+.1f}%")
        
        print("\nComputation Speed:")
        times = [
            ('Electrical', elec['compute_time_ms']),
            ('Dijkstra', dijk['compute_time_ms']),
            ('OSPF', ospf['compute_time_ms']),
            ('ECMP', ecmp['compute_time_ms'])
        ]
        fastest = min(t[1] for t in times if t[1] > 0)
        for name, time in times:
            if time > 0:
                print(f"   {name:<12} {time:>8.3f} ms  ({time/fastest:.1f}x slowest)")
        
        print("\nLoad Balancing Capabilities:")
        for name, result in [('Electrical', elec), ('ECMP', ecmp)]:
            if result['load_balanced']:
                paths = result.get('num_paths', 0)
                print(f"{name}: {paths} parallel paths")
        print(f"Dijkstra: Single path only")
        print(f"OSPF: Single path only")
        
        print("\nBest Algorithm by Metric:")
        print(f"Lowest latency:     Dijkstra ({dijk['latency_ms']:.2f} ms)")
        print(f"Most paths:         "
              f"{'Electrical' if elec['num_paths'] >= ecmp['num_paths'] else 'ECMP'} "
              f"({max(elec['num_paths'], ecmp['num_paths'])} paths)")
        print(f"   Fastest compute:    {min(times, key=lambda x: x[1])[0]} "
              f"({min(times, key=lambda x: x[1])[1]:.3f} ms)")
        
        # Load Balancing Comparison
        print("LOAD BALANCING ANALYSIS")
        
        lb = elec.get('load_balance_metrics', {})
        
        if lb and lb.get('active_edges', 0) > 0:
            print("\nElectrical Flow Load Distribution:")
            print(f"   Active edges:           {lb['active_edges']} of {lb['total_edges']} total")
            print(f"   Effective paths:        {lb['effective_paths']:.2f}")
            print(f"   Gini coefficient:       {lb['gini_coefficient']:.3f} (0=perfect, 1=concentrated)")
            print(f"   Normalized entropy:     {lb['normalized_entropy']:.3f} (0=concentrated, 1=uniform)")
            print(f"   Max/Avg flow ratio:     {lb['max_avg_ratio']:.2f}x")
            print(f"   Distribution quality:   {lb['flow_distribution']}")
            
            # Compare effective paths
            print(f"\nMulti-Path Comparison:")
            print(f"   Electrical Flow:  {lb['effective_paths']:.2f} effective paths")
            print(f"   ECMP:             {ecmp['num_paths']} equal-cost paths")
            print(f"   Dijkstra:         1.00 single path")
            print(f"   OSPF:             1.00 single path")


def visualize_routing_results(router: ElectricalFlowRouter, comparison: Dict, output_dir: str = "routing_visualizations"):
    """
    Generate visualizations of routing comparison.
    Saves multiple graph files to output directory.
    Args:
        router: ElectricalFlowRouter instance
        comparison: Comparison results dictionary
        output_dir: Directory to save visualization files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating visualizations in '{output_dir}/'...")
    
    G = router.G
    elec = comparison['electrical']
    dijk = comparison['dijkstra']
    ospf = comparison['ospf']
    ecmp = comparison['ecmp']
    
    # Get positions for all nodes (use spring layout for 2D projection)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Network Topology Overview
    plt.figure(figsize=(14, 10))
    plt.title("LEO Satellite Network Topology", fontsize=16, fontweight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300, alpha=0.8)
    source = comparison['electrical']['paths'][0][0] if elec['paths'] else None
    dest = comparison['electrical']['paths'][0][-1] if elec['paths'] else None
    if source is not None and dest is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[source], 
                              node_color='green', node_size=600, 
                              label='Source', alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=[dest], 
                              node_color='red', node_size=600,
                              label='Destination', alpha=0.9)
    
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.legend(loc='upper right', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_network_topology.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 1_network_topology.png")
    
    # Dijkstra Shortest Path
    if dijk['path']:
        plt.figure(figsize=(14, 10))
        plt.title("Dijkstra Shortest Path Routing", fontsize=16, fontweight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.2, width=0.5)
        path_edges = [(dijk['path'][i], dijk['path'][i+1]) for i in range(len(dijk['path'])-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                              edge_color='blue', width=3, alpha=0.8,
                              arrows=True, arrowsize=20)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=200, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=dijk['path'], 
                              node_color='blue', node_size=400, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=[source], 
                              node_color='green', node_size=600)
        nx.draw_networkx_nodes(G, pos, nodelist=[dest], 
                              node_color='red', node_size=600)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        stats_text = f"Path Length: {len(dijk['path'])-1} hops\n"
        stats_text += f"Latency: {dijk['latency_ms']:.2f} ms\n"
        stats_text += f"Capacity: {dijk['capacity_gbps']:.2f} Gbps\n"
        stats_text += f"Compute Time: {dijk['compute_time_ms']:.3f} ms"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_dijkstra_path.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: 2_dijkstra_path.png")
    
    # OSPF Routing
    if ospf['path']:
        plt.figure(figsize=(14, 10))
        plt.title("OSPF Bandwidth-Aware Routing", fontsize=16, fontweight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.2, width=0.5)
        path_edges = [(ospf['path'][i], ospf['path'][i+1]) for i in range(len(ospf['path'])-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                              edge_color='purple', width=3, alpha=0.8,
                              arrows=True, arrowsize=20)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=200, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=ospf['path'], 
                              node_color='purple', node_size=400, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=[source], 
                              node_color='green', node_size=600)
        nx.draw_networkx_nodes(G, pos, nodelist=[dest], 
                              node_color='red', node_size=600)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        stats_text = f"Path Length: {ospf['path_length']} hops\n"
        stats_text += f"Latency: {ospf['latency_ms']:.2f} ms\n"
        stats_text += f"Capacity: {ospf['capacity_gbps']:.2f} Gbps\n"
        stats_text += f"OSPF Cost: {ospf['total_cost']:.3f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_ospf_routing.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: 3_ospf_routing.png")
    
    # ECMP Multi-Path
    if ecmp['paths']:
        plt.figure(figsize=(14, 10))
        plt.title("ECMP Equal-Cost Multi-Path Routing", fontsize=16, fontweight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.2, width=0.5)
        colors = ['orange', 'brown', 'pink', 'cyan']
        all_path_nodes = set()
        for idx, path in enumerate(ecmp['paths'][:4]):
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            color = colors[idx % len(colors)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                  edge_color=color, width=2.5, alpha=0.7,
                                  arrows=True, arrowsize=15,
                                  label=f"Path {idx+1}" if ecmp['num_paths'] > 1 else None)
            all_path_nodes.update(path)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=200, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=list(all_path_nodes), 
                              node_color='orange', node_size=400, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=[source], 
                              node_color='green', node_size=600)
        nx.draw_networkx_nodes(G, pos, nodelist=[dest], 
                              node_color='red', node_size=600)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        stats_text = f"Number of Paths: {ecmp['num_paths']}\n"
        stats_text += f"Avg Latency: {ecmp['avg_latency_ms']:.2f} ms\n"
        stats_text += f"Avg Capacity: {ecmp['avg_capacity_gbps']:.2f} Gbps\n"
        stats_text += f"Load Balanced: {'Yes' if ecmp['load_balanced'] else 'No'}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        if ecmp['num_paths'] > 1:
            plt.legend(loc='upper right', fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/4_ecmp_multipath.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: 4_ecmp_multipath.png")
    
    # Electrical Flow - Multi-Path with Flow Intensity
    if elec['paths']:
        plt.figure(figsize=(14, 10))
        plt.title("Electrical Flow Routing (Physics-Based Load Balancing)", 
                 fontsize=16, fontweight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.2, width=0.5)
        colors = ['red', 'darkorange', 'gold', 'yellow', 'lightyellow']
        for idx, path in enumerate(elec['paths'][:5]):
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            # Width decreases for lower-priority paths
            width = 4 - (idx * 0.6)
            alpha = 0.9 - (idx * 0.15)
            color = colors[min(idx, len(colors)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                  edge_color=color, width=max(width, 1), alpha=alpha,
                                  arrows=True, arrowsize=15,
                                  label=f"Path {idx+1}" if elec['num_paths'] > 1 else None)
        all_path_nodes = set()
        for path in elec['paths']:
            all_path_nodes.update(path)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=200, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=list(all_path_nodes), 
                              node_color='yellow', node_size=400, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=[source], 
                              node_color='green', node_size=600)
        nx.draw_networkx_nodes(G, pos, nodelist=[dest], 
                              node_color='red', node_size=600)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        lb = elec.get('load_balance_metrics', {})
        stats_text = f"Parallel Paths: {elec['num_paths']}\n"
        stats_text += f"Primary Latency: {elec['primary_latency_ms']:.2f} ms\n"
        stats_text += f"Primary Capacity: {elec['primary_capacity_gbps']:.2f} Gbps\n"
        if lb:
            stats_text += f"\nLoad Balance:\n"
            stats_text += f"  Gini: {lb.get('gini_coefficient', 0):.3f}\n"
            stats_text += f"  Effective Paths: {lb.get('effective_paths', 0):.2f}\n"
            stats_text += f"  {lb.get('flow_distribution', 'N/A')}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        
        if elec['num_paths'] > 1:
            plt.legend(loc='upper right', fontsize=9)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/5_electrical_flow_multipath.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: 5_electrical_flow_multipath.png")
    # Algorithm Comparison - Bar Charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Routing Algorithm Performance Comparison", fontsize=18, fontweight='bold')
    
    algorithms = ['Electrical', 'Dijkstra', 'OSPF', 'ECMP']
    colors_algo = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']
    
    # Subplot 1: Latency Comparison
    ax1 = axes[0, 0]
    latencies = [
        elec['primary_latency_ms'] if elec['primary_latency_ms'] != float('inf') else 0,
        dijk['latency_ms'] if dijk['latency_ms'] != float('inf') else 0,
        ospf['latency_ms'] if ospf['latency_ms'] != float('inf') else 0,
        ecmp['avg_latency_ms'] if ecmp['avg_latency_ms'] != float('inf') else 0
    ]
    bars1 = ax1.bar(algorithms, latencies, color=colors_algo, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('End-to-End Latency', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, lat in zip(bars1, latencies):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{lat:.1f}ms', ha='center', va='bottom', fontsize=10)
    
    # Subplot 2: Number of Paths
    ax2 = axes[0, 1]
    num_paths = [elec['num_paths'], 1, 1, ecmp['num_paths']]
    bars2 = ax2.bar(algorithms, num_paths, color=colors_algo, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Paths', fontsize=12, fontweight='bold')
    ax2.set_title('Multi-Path Capability', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, paths in zip(bars2, num_paths):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(paths)}', ha='center', va='bottom', fontsize=10)
    
    # Subplot 3: Computation Time
    ax3 = axes[1, 0]
    compute_times = [
        elec['compute_time_ms'],
        dijk['compute_time_ms'],
        ospf['compute_time_ms'],
        ecmp['compute_time_ms']
    ]
    bars3 = ax3.bar(algorithms, compute_times, color=colors_algo, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Computation Time (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Algorithm Speed', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, time in zip(bars3, compute_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}ms', ha='center', va='bottom', fontsize=9)
    
    # Subplot 4: Load Balancing Score (Electrical Flow only)
    ax4 = axes[1, 1]
    lb = elec.get('load_balance_metrics', {})
    if lb and lb.get('active_edges', 0) > 0:
        # Normalize all metrics to 0-1 scale for fair comparison
        metrics = ['Normalized\nEntropy', 'Effective\nPaths\n(norm.)', '1 - Gini\nCoeff']
        
        # Normalize effective paths: cap at reasonable value and scale
        effective_paths_raw = lb.get('effective_paths', 0)
        effective_paths_normalized = min(effective_paths_raw / 10.0, 1.0)  # Cap at 10 paths = 1.0
        
        values = [
            lb.get('normalized_entropy', 0),
            effective_paths_normalized,  # Now properly normalized
            1 - lb.get('gini_coefficient', 1)
        ]
        bars4 = ax4.bar(metrics, values, color=['#FF6B6B', '#FF8E8E', '#FFB0B0'], 
                       alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
        ax4.set_title('Electrical Flow Load Balancing Metrics', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 1.1)
        ax4.grid(axis='y', alpha=0.3)
        ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good Threshold')
        ax4.legend(fontsize=8)
        
        # Add value labels with actual effective paths count in parentheses
        for i, (bar, val) in enumerate(zip(bars4, values)):
            height = bar.get_height()
            if i == 1:  # Effective paths metric
                label_text = f'{val:.3f}\n({effective_paths_raw:.1f} paths)'
            else:
                label_text = f'{val:.3f}'
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    label_text, ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No Load Balance\nData Available', 
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/6_algorithm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 6_algorithm_comparison.png")
    
    # Load Distribution Heatmap (Electrical Flow)
    if lb and lb.get('active_edges', 0) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Electrical Flow - Load Distribution Analysis", 
                    fontsize=16, fontweight='bold')
        # Get edge flows
        edge_flows = comparison.get('edge_flows', {})
        if edge_flows:
            flows = np.array([abs(f) for f in edge_flows.values()])
            flows = flows[flows > 1e-9]
            flows_sorted = np.sort(flows)[::-1]  # Descending
            # Left plot: Flow distribution
            ax1.plot(range(1, len(flows_sorted)+1), flows_sorted, 
                    'o-', color='#FF6B6B', linewidth=2, markersize=6)
            ax1.set_xlabel('Edge Rank', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Flow Magnitude', fontsize=12, fontweight='bold')
            ax1.set_title('Flow Distribution Across Active Edges', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            # Right plot: Load balance metrics radar-style
            metrics_names = ['Entropy', 'Uniform\nDistribution', 'Path\nDiversity']
            metrics_values = [
                lb.get('normalized_entropy', 0),
                1 - lb.get('gini_coefficient', 1),
                min(lb.get('effective_paths', 0) / 5, 1.0)  # Normalize to 0-1
            ]
            
            x = np.arange(len(metrics_names))
            bars = ax2.barh(x, metrics_values, color=['#4ECDC4', '#95E1D3', '#FFA07A'], 
                           alpha=0.7, edgecolor='black')
            ax2.set_yticks(x)
            ax2.set_yticklabels(metrics_names)
            ax2.set_xlabel('Score (0-1)', fontsize=12, fontweight='bold')
            ax2.set_title('Load Balancing Quality Metrics', fontsize=14)
            ax2.set_xlim(0, 1.1)
            ax2.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, 
                       label='Good Threshold')
            ax2.legend()
            ax2.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, metrics_values)):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2.,
                        f' {val:.3f}', ha='left', va='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/7_load_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: 7_load_distribution.png")
    
    print(f"\nAll visualizations saved to '{output_dir}/' directory!")


def demo_comprehensive_routing(num_satellites=100, max_isl_range_km=5000):
    """
    Demonstrate comprehensive routing algorithm comparison.
    
    Args:
        num_satellites: Number of satellites to simulate (30-1000+)
        max_isl_range_km: Maximum inter-satellite link range [km]
    """
    
    # Load constellation
    print(f"Loading LEO constellation ({num_satellites} satellites)...")
    constellation = LEOConstellation(max_satellites=num_satellites)
    
    # Build topology
    print(f"\nBuilding network topology (max range: {max_isl_range_km} km)...")
    current_time = datetime.utcnow()
    G = constellation.build_network_topology(current_time, max_distance_km=max_isl_range_km)
    
    if G.number_of_edges() == 0:
        print("Error: No viable links in constellation")
        return
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Initialize router
    print("\nStep 3: Initializing routing engine...")
    
    # TUNING PARAMETERS 
    ALPHA = 3.0  # Latency weight (higher = prefer low latency paths)
    BETA = 0.3   # Capacity weight (higher = prefer high capacity paths)
    
    router = ElectricalFlowRouter(G, alpha=ALPHA, beta=BETA)
    
    # Verify router has enough nodes
    if router.n_nodes < 2:
        print("Error: Router needs at least 2 connected nodes")
        return None, None
    
    # Select source and destination - find connected nodes
    print("\nFinding connected node pairs...")
    
    # Build a subgraph with only the nodes that have edges (router's node list)
    router_nodes = set(router.node_list)
    G_filtered = G.subgraph(router_nodes).copy()
    
    print(f"Router has {len(router_nodes)} nodes with edges")
    print(f"Filtered graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    
    if G_filtered.number_of_nodes() < 2:
        print("Error: Not enough connected nodes")
        return None, None
    # Find largest connected component in the filtered graph
    # For directed graphs, use weakly connected (ignores direction)
    # since ISL links are bidirectional in practice
    if G_filtered.is_directed():
        components = list(nx.weakly_connected_components(G_filtered))
    else:
        components = list(nx.connected_components(G_filtered))
    
    # Use the largest component
    largest_component = max(components, key=len)
    component_nodes = sorted(list(largest_component))
    
    print(f"Found largest connected component: {len(component_nodes)} nodes")
    print(f"Component nodes: {component_nodes}")
    
    if len(component_nodes) < 2:
        print("Error: Largest component has fewer than 2 nodes")
        return None, None
    
    # Select nodes from opposite ends for interesting multi-hop routes
    source = component_nodes[0]
    dest = component_nodes[-1]
    
    # CRITICAL: Verify path exists in FILTERED graph (not original G)
    try:
        test_path = nx.shortest_path(G_filtered, source, dest, weight='latency')
        path_length = len(test_path) - 1
        print(f"Route verified: {source} -> {dest} ({path_length} hops)")
        print(f"   Path: {' -> '.join(map(str, test_path))}")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        print(f"   Path verification failed for {source} → {dest}: {e}")
        print(f"   Source in component: {source in component_nodes}")
        print(f"   Dest in component: {dest in component_nodes}")
        print(f"   Source in G_filtered: {source in G_filtered.nodes()}")
        print(f"   Dest in G_filtered: {dest in G_filtered.nodes()}")
        
        # Try finding any two connected nodes in the component
        print("   Searching for alternative connected pair...")
        found = False
        
        # Try all pairs systematically
        for i, src in enumerate(component_nodes):
            for dst in component_nodes[i+1:]:
                try:
                    test_path = nx.shortest_path(G_filtered, src, dst, weight='latency')
                    path_length = len(test_path) - 1
                    source = src
                    dest = dst
                    print(f"Found valid route: {source} -> {dest} ({path_length} hops)")
                    print(f"   Path: {' -> '.join(map(str, test_path))}")
                    found = True
                    break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            if found:
                break
        
        if not found:
            print("Error: Could not find any valid path in component")

            print(f"   Edges in filtered graph involving component nodes:")
            component_edges = [(u, v) for u, v in G_filtered.edges() 
                             if u in component_nodes and v in component_nodes]
            print(f"   Found {len(component_edges)} edges within component")
            if len(component_edges) < 20:
                print(f"   Component edges: {component_edges}")
            return None, None
    
    # Compare all algorithms
    print("\nRunning all routing algorithms...")
    comparison = router.compare_all_algorithms(source, dest)
    
    if comparison:
        router.print_comprehensive_comparison(comparison)
        
        # Generate visualizations
        visualize_routing_results(router, comparison)
        
        # Additional diagnostics
        print("\nADDITIONAL METRICS")
        
        elec = comparison['electrical']
        dijk = comparison['dijkstra']
        ospf = comparison['ospf']
        ecmp = comparison['ecmp']
        
        # Show actual paths if available
        print("\nRouting Paths:")
        if dijk['path']:
            print(f"\n   Dijkstra path ({len(dijk['path'])-1} hops):")
            print(f"   {' -> '.join(map(str, dijk['path']))}")
        
        if ospf['path']:
            print(f"\n   OSPF path ({len(ospf['path'])-1} hops):")
            print(f"   {' -> '.join(map(str, ospf['path']))}")
        
        if elec['num_paths'] > 0:
            print(f"\n   Electrical Flow paths ({elec['num_paths']} parallel):")
            for i, path in enumerate(elec['paths'][:3], 1):
                print(f"   Path {i}: {' -> '.join(map(str, path))}")
            
            # Show flow distribution if available
            if 'load_balance_metrics' in elec:
                lb = elec['load_balance_metrics']
                if lb.get('active_edges', 0) > 0:
                    print(f"\n   Flow Distribution:")
                    print(f"   {'█' * int(lb['normalized_entropy'] * 50)} {lb['normalized_entropy']:.1%} uniform")
                    print(f"   Gini: {lb['gini_coefficient']:.3f} | Effective paths: {lb['effective_paths']:.2f}")
        
        if ecmp['num_paths'] > 0:
            print(f"\n   ECMP paths ({ecmp['num_paths']} equal-cost):")
            for i, path in enumerate(ecmp['paths'][:3], 1):
                print(f"   Path {i}: {' -> '.join(map(str, path))}")
        
        # Network statistics
        print(f"\n\nNetwork Statistics:")
        print(f"   Total nodes:              {G.number_of_nodes()}")
        print(f"   Total edges:              {G.number_of_edges()}")
        print(f"   Network diameter:         {nx.diameter(G.to_undirected()) if nx.is_connected(G.to_undirected()) else 'N/A'}")
        print(f"   Average degree:           {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        print(f"   Largest component:        {len(component_nodes)} nodes")
    
    return router, comparison


if __name__ == "__main__":
    # Configure constellation size and parameters
    NUM_SATELLITES = 100      # Increase this: 30, 100, 200, 500, 1000+
    MAX_ISL_RANGE_KM = 5000   # Inter-satellite link range
    
    print("CONFIGURATION:")
    print(f"  Satellites:        {NUM_SATELLITES}")
    print(f"  Max ISL Range:     {MAX_ISL_RANGE_KM} km")
    
    router, comparison = demo_comprehensive_routing(
        num_satellites=NUM_SATELLITES,
        max_isl_range_km=MAX_ISL_RANGE_KM
    )