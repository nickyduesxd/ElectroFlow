from leo_constellation import LEOConstellation
from leo_electrical_routing import ElectricalFlowRouter
from datetime import datetime
import networkx as nx

def test_routing():
    """Test electrical routing with guaranteed connected nodes."""
    
    # Load constellation
    print("Loading LEO constellation...")
    constellation = LEOConstellation(max_satellites=50)
    
    # Build topology
    print("Building network topology...")
    current_time = datetime.utcnow()
    G = constellation.build_network_topology(current_time, max_distance_km=3000)
    
    print(f"Topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    if G.number_of_edges() == 0:
        print("\n⚠ No links found! Try increasing max_distance_km")
        return
    
    # Find a connected pair of nodes
    print("\nFinding connected node pair...")
    
    # Get largest connected component
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    components = list(nx.connected_components(G_undirected))
    largest_component = max(components, key=len)
    
    print(f"Largest connected component: {len(largest_component)} nodes")
    
    if len(largest_component) < 2:
        print("\n⚠ Need at least 2 connected nodes")
        return
    
    # Pick source and dest from largest component
    component_nodes = list(largest_component)
    source = component_nodes[0]
    dest = component_nodes[-1]
    
    print(f"Testing route: {source} → {dest}")
    
    # Initialize router
    router = ElectricalFlowRouter(G)
    
    # Compare algorithms
    print("Running routing algorithms...")
    
    comparison = router.compare_with_dijkstra(source, dest)
    
    if comparison:
        router.print_comparison(comparison)
        
        # Show some path details
        if comparison['electrical']['paths']:
            for i, path in enumerate(comparison['electrical']['paths'][:3], 1):
                print(f"  Path {i}: {' → '.join(map(str, path))}")
        
        if comparison['dijkstra']['path']:
            print(f"  {' → '.join(map(str, comparison['dijkstra']['path']))}")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70 + "\n")
    
    return router, comparison


if __name__ == "__main__":
    router, comparison = test_routing()
    

