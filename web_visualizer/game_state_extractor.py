"""
Extract and serialize Catan game state from catanatron game objects
"""
from catanatron.models.map import NUM_NODES
import json
import math

def get_tile_type_name(resource):
    """Convert resource to readable name"""
    if resource is None:
        return "DESERT"
    return str(resource)

def get_building_name(building_type):
    """Convert building type to readable name"""
    if building_type is None:
        return None
    return str(building_type)

def calculate_node_position(tile_coord, node_ref):
    """
    Calculate pixel position for a node based on its tile and position
    NodeRef positions: NORTH, NORTHEAST, SOUTHEAST, SOUTH, SOUTHWEST, NORTHWEST
    
    The JS draws hexagons with vertices at: -30°, 30°, 90°, 150°, 210°, 270°
    This creates a pointy-top hex rotated -30°
    """
    q, r, s = tile_coord
    
    # Hex radius (must match frontend HEX_RADIUS = 60)
    HEX_RADIUS = 60
    
    # Base tile position - MUST match JS axialToPixel exactly
    # Catanatron Coordinate System (Pointy Top)
    # EAST (1, -1, 0) -> 0 degrees (Right)
    # NORTHEAST (1, 0, -1) -> -60 degrees (Top Right)
    # Derived formula:
    # x = radius * sqrt(3)/2 * (q - r)
    # y = radius * -3/2 * (q + r)
    tile_x = HEX_RADIUS * (math.sqrt(3)/2 * (q - r))
    tile_y = HEX_RADIUS * (-1.5 * (q + r))
    
    # Map NodeRef to hex vertex angles
    # Hexagon vertices (from JS): -30°, 30°, 90°, 150°, 210°, 270°
    # These correspond to: NE, E/SE, S, SW, W/NW, N in a rotated layout
    node_ref_str = str(node_ref).split('.')[-1]
    
    # Match catanatron's node positions to hex vertices
    angles = {
        'NORTH': 270,        # Top vertex
        'NORTHEAST': -30,    # Top-right vertex  
        'SOUTHEAST': 30,     # Right vertex (or bottom-right)
        'SOUTH': 90,         # Bottom vertex
        'SOUTHWEST': 150,    # Bottom-left vertex
        'NORTHWEST': 210,    # Left vertex (or top-left)
    }
    
    angle = angles.get(node_ref_str, 0)
    angle_rad = math.radians(angle)
    
    # Vertices are exactly at HEX_RADIUS from center
    node_x = tile_x + HEX_RADIUS * math.cos(angle_rad)
    node_y = tile_y + HEX_RADIUS * math.sin(angle_rad)
    
    return [node_x, node_y, 0]

def build_node_position_map(game):
    """Build a mapping of node_id to approximate position based on adjacent tiles"""
    board = game.state.board
    node_positions = {}
    node_position_samples = {}  # Track multiple samples per node to average
    
    for tile_coord, tile in board.map.land_tiles.items():
        for node_ref, node_id in tile.nodes.items():
            # Calculate position for this node from this tile's perspective
            pos = calculate_node_position(tile_coord, node_ref)
            
            if node_id not in node_position_samples:
                node_position_samples[node_id] = []
            node_position_samples[node_id].append(pos)
    
    # Average all samples for each node to get final position
    for node_id, samples in node_position_samples.items():
        if samples:
            avg_x = sum(s[0] for s in samples) / len(samples)
            avg_y = sum(s[1] for s in samples) / len(samples)
            node_positions[node_id] = [avg_x, avg_y, 0]
    
    return node_positions


def extract_edges(game, node_positions):
    """Extract roads from the board"""
    board = game.state.board
    edges = []
    
    # In catanatron, edge_id is a tuple of (node_id1, node_id2)
    # So we can directly use it to get the positions!
    for edge_id, color in board.roads.items():
        # edge_id is a tuple like (0, 1) representing the two nodes
        if isinstance(edge_id, tuple) and len(edge_id) == 2:
            node_id1, node_id2 = edge_id
            coord1 = node_positions.get(node_id1)
            coord2 = node_positions.get(node_id2)
            
            # Only add the road if both nodes have valid positions
            if coord1 is None or coord2 is None:
                print(f"Warning: Missing node position for edge {edge_id}")
                continue
                
            edge_data = {
                'edge_id': str(edge_id),
                'color': color.name,
                'coordinates': [coord1, coord2]
            }
            edges.append(edge_data)
    
    return edges

def extract_board_tiles(game):
    """Extract the hex tiles from the game board"""
    board = game.state.board
    tiles = []
    
    for coordinate, tile in board.map.land_tiles.items():
        tile_data = {
            'coordinate': list(coordinate),  # (q, r, s) axial coordinates
            'resource': get_tile_type_name(tile.resource),
            'number': tile.number if hasattr(tile, 'number') else None,
            'has_robber': coordinate == board.robber_coordinate
        }
        tiles.append(tile_data)
    
    return tiles

def extract_nodes(game, node_positions):
    """Extract settlements and cities from the board"""
    board = game.state.board
    nodes = []
    
    for node_id, building in board.buildings.items():
        # Building is a tuple: (color, building_type)
        color, building_type = building
        node_data = {
            'node_id': node_id,
            'building_type': str(building_type),
            'color': color.name,
            'coordinate': node_positions.get(node_id, [0, 0, 0])
        }
        nodes.append(node_data)
    
    return nodes

def extract_edges(game, node_positions):
    """Extract roads from the board"""
    board = game.state.board
    edges = []
    
    # In catanatron, edge_id is actually a tuple of (node_id1, node_id2)
    # So we can directly use it to get the positions!
    for edge_id, color in board.roads.items():
        # edge_id is a tuple like (0, 1) representing the two nodes
        if isinstance(edge_id, tuple) and len(edge_id) == 2:
            node_id1, node_id2 = edge_id
            coord1 = node_positions.get(node_id1, [0, 0, 0])
            coord2 = node_positions.get(node_id2, [0, 0, 0])
        else:
            coord1 = coord2 = [0, 0, 0]
            
        edge_data = {
            'edge_id': str(edge_id),
            'color': color.name,
            'coordinates': [coord1, coord2]
        }
        edges.append(edge_data)
    
    return edges

def extract_players(game):
    """Extract player information"""
    players = []
    
    for player in game.state.players:
        # Get resource counts by iterating through player state
        resources = {}
        resource_types = ['WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE']
        for res_type in resource_types:
            resources[res_type] = 0
        
        # Count resources from player's hand
        player_state = game.state.player_state
        player_key = f"P{game.state.color_to_index[player.color]}_"
        
        # Try to get resource counts from state keys
        for res in resource_types:
            key = f"{player_key}{res}_IN_HAND"
            if key in player_state:
                resources[res] = player_state[key]
        
        # Calculate dev cards in hand
        dev_card_types = [
            'KNIGHT', 'YEAR_OF_PLENTY', 'MONOPOLY', 
            'ROAD_BUILDING', 'VICTORY_POINT'
        ]
        dev_cards_count = sum(
            player_state.get(f"{player_key}{dc}_IN_HAND", 0) 
            for dc in dev_card_types
        )

        player_data = {
            'name': player.color.name,
            'color': player.color.name,
            'victory_points': player_state.get(f"{player_key}VICTORY_POINTS", 0),
            'resources': resources,
            'dev_cards': dev_cards_count,
            'longest_road': player_state.get(f"{player_key}HAS_ROAD", False),
            'largest_army': player_state.get(f"{player_key}HAS_ARMY", False),
            'roads_left': player_state.get(f"{player_key}ROADS_AVAILABLE", 15),
            'settlements_left': player_state.get(f"{player_key}SETTLEMENTS_AVAILABLE", 5),
            'cities_left': player_state.get(f"{player_key}CITIES_AVAILABLE", 4)
        }
        players.append(player_data)
    
    return players

def extract_ports(game, node_positions):
    """Extract port locations and types"""
    board = game.state.board
    ports = []
    
    # port_nodes is a dict of resource -> set of node_ids
    for resource, node_ids in board.map.port_nodes.items():
        for node_id in node_ids:
            port_data = {
                'node_id': node_id,
                'resource': str(resource) if resource else "THREE_TO_ONE",
                'coordinate': node_positions.get(node_id, [0, 0, 0])
            }
            ports.append(port_data)
    
    return ports

def extract_game_state(game, episode=0, step=0, action_taken=None):
    """
    Extract complete game state for visualization
    
    Args:
        game: Catanatron Game object
        episode: Current episode number
        step: Current step in episode
        action_taken: Description of last action taken
    
    Returns:
        Dictionary containing complete game state
    """
    try:
        # Build node position mapping first
        node_positions = build_node_position_map(game)
        
        # Extract last dice roll from actions
        last_dice_roll = None
        if len(game.state.actions) > 0:
            last_action = game.state.actions[-1]
            action_str = str(last_action)
            if 'ROLL' in action_str:
                # Extract dice values from action string like "Action(BLUE ROLL (3, 4))"
                import re
                match = re.search(r'ROLL \((\d+), (\d+)\)', action_str)
                if match:
                    last_dice_roll = [int(match.group(1)), int(match.group(2))]
        
        state = {
            'episode': episode,
            'step': step,
            'current_player': game.state.current_player().color.name,
            'tiles': extract_board_tiles(game),
            'nodes': extract_nodes(game, node_positions),
            'edges': extract_edges(game, node_positions),
            'players': extract_players(game),
            'ports': extract_ports(game, node_positions),
            'action': action_taken,
            'dice_roll': last_dice_roll,
            'game_over': game.winning_color() is not None,
            'winner': game.winning_color().name if game.winning_color() else None
        }
        return state
    except Exception as e:
        print(f"Error extracting game state: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'episode': episode,
            'step': step
        }
