# --- Imports ---
import math
import matplotlib.pyplot as plt
import itertools
import random
import numpy as np
import time
from copy import deepcopy
import multiprocessing  # Added for parallelism

# DEAP_AVAILABLE = False

# Try importing DEAP
try:
    from deap import base, creator, tools, algorithms

    DEAP_AVAILABLE = True
except ImportError:
    print(
        "WARNING: DEAP library not found. Genetic Algorithm optimization will not be available."
    )
    print("Install using: pip install deap")
    DEAP_AVAILABLE = False

# Try importing Numba
try:
    from numba import jit

    NUMBA_AVAILABLE = True
    print("Numba detected, JIT compilation enabled.")
except ImportError:
    print("WARNING: Numba library not found. JIT compilation will be disabled.")
    print("Install using: pip install numba")
    NUMBA_AVAILABLE = False

# --- Configuration (Keep as before) ---
# Area and Plot Setup
AREA_WIDTH_M = 1000
AREA_HEIGHT_M = 600
PLOT_SIZE_M = 100
NUM_PLOTS_X = AREA_WIDTH_M // PLOT_SIZE_M
NUM_PLOTS_Y = AREA_HEIGHT_M // PLOT_SIZE_M
# Node Ranges
SENSOR_PERFECT_RANGE_M = 60
SENSOR_MAX_RANGE_M = 60
CAMERA_RANGE_M = 200
COMMUNICATION_RANGE_M = 60
# Base Station
BASE_STATION_POS = (0, AREA_HEIGHT_M / 2)
BASE_STATION_ID = "BS_0"
# Node Types
NODE_TYPES = {
    "SENSOR": {"marker": "o", "color": "blue", "label": "Sensor/PIR", "cost": 1.0},
    "CAMERA": {"marker": "s", "color": "red", "label": "Camera", "cost": 3.0},
    "RELAY": {"marker": "x", "color": "green", "label": "Relay", "cost": 0.5},
    "BASE_STATION": {
        "marker": "*",
        "color": "black",
        "label": "Base Station",
        "size": 200,
        "cost": 0,
    },
}
# --- GA Config (Keep as before or adjust) ---
POTENTIAL_SITE_GRID_STEP = 15
COVERAGE_TARGET_STEP = 10
WEIGHT_SENSOR_COVERAGE = 1000.0
WEIGHT_CAMERA_COVERAGE = 450.0
WEIGHT_CONNECTIVITY_PENALTY = 2500.0
WEIGHT_COST = 0.9
MIN_ACCEPTABLE_SENSOR_COVERAGE = 1.0
SENSOR_COVERAGE_PENALTY = 850.0
POPULATION_SIZE = 750
NUM_GENERATIONS = 160
CROSSOVER_PROB = 0.35
MUTATION_PROB = 0.0225  # Maybe slightly lower mutation

# --- Utility Functions ---
# **** MODIFICATION: Add Numba JIT ****
if NUMBA_AVAILABLE:

    @jit(nopython=True, fastmath=True)  # fastmath=True can give extra speed
    def dist(p1, p2):
        """Calculate Euclidean distance (JIT enabled)."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)
else:
    # Fallback pure Python version if Numba is not available
    def dist(p1, p2):
        """Calculate Euclidean distance."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# --- Heuristic Placement (Keep as before) ---
def place_heuristic_nodes():
    """Places nodes based on the initial heuristic strategy."""
    nodes = []
    # Sensors in plot centers
    s_idx = 0
    for i in range(NUM_PLOTS_X):
        for j in range(NUM_PLOTS_Y):
            x = (i + 0.5) * PLOT_SIZE_M
            y = (j + 0.5) * PLOT_SIZE_M
            nodes.append({"id": f"S_{s_idx}", "type": "SENSOR", "pos": (x, y)})
            s_idx += 1

    # Relays at midpoints between sensors
    r_idx = 0
    relay_coords = set()
    # Horizontal relays
    for j in range(NUM_PLOTS_Y):
        for i in range(NUM_PLOTS_X - 1):
            x = (i + 1) * PLOT_SIZE_M
            y = (j + 0.5) * PLOT_SIZE_M
            relay_coords.add((x, y))
    # Vertical relays
    for i in range(NUM_PLOTS_X):
        for j in range(NUM_PLOTS_Y - 1):
            x = (i + 0.5) * PLOT_SIZE_M
            y = (j + 1) * PLOT_SIZE_M
            relay_coords.add((x, y))
    # Intersection relays (optional, for robustness)
    for i in range(1, NUM_PLOTS_X):
        for j in range(1, NUM_PLOTS_Y):
            x = i * PLOT_SIZE_M
            y = j * PLOT_SIZE_M
            relay_coords.add((x, y))

    for pos in relay_coords:
        nodes.append({"id": f"R_{r_idx}", "type": "RELAY", "pos": pos})
        r_idx += 1

    # Cameras (strategic placement from previous example)
    cam_idx = 0
    camera_locations = [
        (AREA_WIDTH_M * 0.25, AREA_HEIGHT_M * 0.25), (AREA_WIDTH_M * 0.75, AREA_HEIGHT_M * 0.25),
        (AREA_WIDTH_M * 0.50, AREA_HEIGHT_M * 0.50), (AREA_WIDTH_M * 0.25, AREA_HEIGHT_M * 0.75),
        (AREA_WIDTH_M * 0.75, AREA_HEIGHT_M * 0.75), (AREA_WIDTH_M * 0.50, AREA_HEIGHT_M * 0.85),
    ]
    for pos in camera_locations:
        nodes.append({"id": f"C_{cam_idx}", "type": "CAMERA", "pos": pos})
        cam_idx += 1

    # Base Station
    nodes.append({"id": BASE_STATION_ID, "type": "BASE_STATION", "pos": BASE_STATION_POS})
    return nodes


# --- Analysis Functions ---
# Connectivity check is complex for JIT, leave as pure Python
def check_connectivity(nodes, comm_range, base_station_id):
    base_station = next((n for n in nodes if n["id"] == base_station_id), None)
    if not base_station or not nodes:
        return 0.0, [n["id"] for n in nodes if n["type"] in ["SENSOR", "CAMERA"]]
    q = [base_station]
    reachable = {base_station_id}
    head = 0
    all_nodes_map = {n["id"]: n for n in nodes}
    while head < len(q):
        current_node = q[head]
        head += 1
        for other_id, other_node in all_nodes_map.items():
            if other_id not in reachable:
                # Uses the potentially jitted dist function
                if dist(current_node["pos"], other_node["pos"]) <= comm_range:
                    reachable.add(other_id)
                    q.append(other_node)
    sensors_cameras = [n for n in nodes if n["type"] in ["SENSOR", "CAMERA"]]
    connected_sc_count = 0
    unconnected_sc_ids = []
    for sc_node in sensors_cameras:
        if sc_node["id"] in reachable:
            connected_sc_count += 1
        else:
            unconnected_sc_ids.append(sc_node["id"])
    total_sc = len(sensors_cameras)
    fraction_connected = (connected_sc_count / total_sc) if total_sc > 0 else 1.0
    return fraction_connected, unconnected_sc_ids


# **** MODIFICATION: Attempt to JIT calculate_coverage ****
# Note: JIT might struggle with lists of dictionaries directly.
# If this fails with Numba errors (especially in nopython mode),
# we would need to refactor it to accept NumPy arrays of positions.
if NUMBA_AVAILABLE:
    # We need to pass primitive types or arrays Numba understands.
    # Let's create a helper function that takes arrays.
    @jit(nopython=True)
    def _calculate_coverage_jit(covering_nodes_pos, target_points_arr, range_m):
        if covering_nodes_pos.shape[0] == 0 or target_points_arr.shape[0] == 0:
            return 0.0

        covered_point_count = 0
        num_target_points = target_points_arr.shape[0]
        num_covering_nodes = covering_nodes_pos.shape[0]

        for i in range(num_target_points):
            tp = (target_points_arr[i, 0], target_points_arr[i, 1])
            for j in range(num_covering_nodes):
                node_pos = (covering_nodes_pos[j, 0], covering_nodes_pos[j, 1])
                # Calculate distance using JITted dist (or inline)
                dx = tp[0] - node_pos[0]
                dy = tp[1] - node_pos[1]
                if (
                    dx * dx + dy * dy
                ) <= range_m * range_m:  # Compare squared distances
                    covered_point_count += 1
                    break  # Move to next target point
        return covered_point_count / num_target_points

    def calculate_coverage(nodes, target_points, range_m, node_type_to_check):
        """Calculates fraction of target points covered (uses JIT helper)."""
        if not target_points:
            return 1.0
        # Extract positions into NumPy arrays for the JIT function
        covering_nodes_pos_list = [
            n["pos"] for n in nodes if n["type"] == node_type_to_check
        ]
        if not covering_nodes_pos_list:
            return 0.0

        covering_nodes_pos_arr = np.array(covering_nodes_pos_list, dtype=np.float64)
        target_points_arr = np.array(target_points, dtype=np.float64)

        return _calculate_coverage_jit(
            covering_nodes_pos_arr, target_points_arr, range_m
        )

else:
    # Original pure Python version if Numba is not available
    def calculate_coverage(nodes, target_points, range_m, node_type_to_check):
        if not target_points:
            return 1.0
        covering_nodes = [n for n in nodes if n["type"] == node_type_to_check]
        if not covering_nodes:
            return 0.0
        covered_points_set = set()
        for tp in target_points:
            for node in covering_nodes:
                if dist(tp, node["pos"]) <= range_m:
                    covered_points_set.add(tp)
                    break
        return len(covered_points_set) / len(target_points)


# **** MODIFICATION: Add JIT to get_covered_points helper ****
if NUMBA_AVAILABLE:

    @jit(nopython=True)
    def _get_covered_points_jit(node_pos_tuple, target_points_arr, range_m):
        """Helper: returns indices of target points covered by a single node."""
        covered_indices = []
        range_sq = range_m * range_m
        node_x, node_y = node_pos_tuple[0], node_pos_tuple[1]
        for i in range(target_points_arr.shape[0]):
            tp_x, tp_y = target_points_arr[i, 0], target_points_arr[i, 1]
            dx = node_x - tp_x
            dy = node_y - tp_y
            if (dx * dx + dy * dy) <= range_sq:
                covered_indices.append(i)
        # Return as a NumPy array for potential future use, though set needed outside
        return np.array(covered_indices, dtype=np.int64)

    def get_covered_points(node, target_points, range_m):
        """Helper: returns the set of target points covered by a single node (uses JIT)."""
        target_points_arr = np.array(target_points, dtype=np.float64)
        node_pos_tuple = (node["pos"][0], node["pos"][1])  # Ensure tuple for JIT
        covered_indices = _get_covered_points_jit(
            node_pos_tuple, target_points_arr, range_m
        )
        # Convert indices back to tuples (or keep as indices if target_points structure allows)
        return {target_points[i] for i in covered_indices}

else:
    # Original pure Python version
    def get_covered_points(node, target_points, range_m):
        return {tp for tp in target_points if dist(node["pos"], tp) <= range_m}


# Keep analyze_deployment and plot_deployment as they were
def analyze_deployment(
    nodes, comm_range, sensor_range, camera_range, target_points, base_station_id
):
    print("\n--- Analysis Results ---")
    if not nodes:
        print("No nodes to analyze.")
        return {}
    num_nodes = len([n for n in nodes if n["type"] != "BASE_STATION"])
    sensors = [n for n in nodes if n["type"] == "SENSOR"]
    cameras = [n for n in nodes if n["type"] == "CAMERA"]
    relays = [n for n in nodes if n["type"] == "RELAY"]
    print(
        f"Node Counts: Total={num_nodes}, Sensors={len(sensors)}, Cameras={len(cameras)}, Relays={len(relays)}"
    )
    total_cost = sum(
        NODE_TYPES[n["type"]]["cost"] for n in nodes if n["type"] != "BASE_STATION"
    )
    print(f"Estimated Cost: {total_cost:.2f}")
    connectivity_fraction, unconnected = check_connectivity(
        nodes, comm_range, base_station_id
    )
    print(
        f"Connectivity (Sensors/Cameras to Base Station): {connectivity_fraction*100:.2f}%"
    )
    if unconnected:
        print(f"  Unconnected Nodes: {sorted(unconnected)}")
    sensor_coverage = calculate_coverage(nodes, target_points, sensor_range, "SENSOR")
    print(f"Sensor Coverage ({sensor_range}m range): {sensor_coverage*100:.2f}%")
    camera_coverage = calculate_coverage(nodes, target_points, camera_range, "CAMERA")
    print(f"Camera Coverage ({camera_range}m range): {camera_coverage*100:.2f}%")
    return {
        "cost": total_cost,
        "connectivity": connectivity_fraction,
        "sensor_coverage": sensor_coverage,
        "camera_coverage": camera_coverage,
    }


def plot_deployment(
    nodes, area_width, area_height, plot_size, title="WSN Deployment Plan", ranges=True
):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-plot_size * 0.1, area_width + plot_size * 0.1)
    ax.set_ylim(-plot_size * 0.1, area_height + plot_size * 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title(title)
    num_plots_x = area_width // plot_size
    num_plots_y = area_height // plot_size
    for i in range(num_plots_x + 1):
        ax.axvline(i * plot_size, color="grey", linestyle="--", linewidth=0.5)
    for j in range(num_plots_y + 1):
        ax.axhline(j * plot_size, color="grey", linestyle="--", linewidth=0.5)
    legend_elements = {}
    for node in nodes:
        node_type = node["type"]
        pos = node["pos"]
        config = NODE_TYPES[node_type]
        s = config.get("size", 100 if node_type != "RELAY" else 60)
        ax.scatter(
            pos[0],
            pos[1],
            marker=config["marker"],
            color=config["color"],
            s=s,
            label=config["label"] if node_type not in legend_elements else "",
            zorder=5,
        )
        legend_elements[node_type] = config
        if ranges:
            range_val = 0
            alpha_val = 0.08
            fill_val = True
            linestyle = "-"
            edgecolor = config["color"]
            if node_type == "SENSOR":
                range_val = SENSOR_MAX_RANGE_M
                linestyle = ":"
                edgecolor = "grey"
            elif node_type == "CAMERA":
                range_val = CAMERA_RANGE_M
            elif node_type in ["RELAY", "BASE_STATION"]:
                range_val = COMMUNICATION_RANGE_M
                fill_val = False
                linestyle = "--"
                alpha_val = 0.3
            if range_val > 0:
                ax.add_patch(
                    plt.Circle(
                        pos,
                        range_val,
                        color=edgecolor,
                        alpha=alpha_val,
                        fill=fill_val,
                        linestyle=linestyle,
                        zorder=1,
                    )
                )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=config["marker"],
            color="w",
            markerfacecolor=config["color"],
            markersize=10,
            label=config["label"],
        )
        for config in legend_elements.values()
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


# --- GA Optimization Components (Requires DEAP) ---
if DEAP_AVAILABLE:
    # --- Discretize Space & Target Points (Keep as before) ---
    potential_sites = []
    site_id_counter = 0
    for x in range(
        POTENTIAL_SITE_GRID_STEP // 2, AREA_WIDTH_M, POTENTIAL_SITE_GRID_STEP
    ):
        for y in range(
            POTENTIAL_SITE_GRID_STEP // 2, AREA_HEIGHT_M, POTENTIAL_SITE_GRID_STEP
        ):
            potential_sites.append({"site_id": site_id_counter, "pos": (x, y)})
            site_id_counter += 1
    num_potential_sites = len(potential_sites)

    target_points = []
    for x in range(COVERAGE_TARGET_STEP // 2, AREA_WIDTH_M, COVERAGE_TARGET_STEP):
        for y in range(COVERAGE_TARGET_STEP // 2, AREA_HEIGHT_M, COVERAGE_TARGET_STEP):
            target_points.append((x, y))
    # Convert target points to NumPy array once for JIT functions
    target_points_global_arr = np.array(target_points, dtype=np.float64)

    # --- Fitness Function (Uses JITted coverage) ---
    def evaluate_deployment_ga(individual):
        placed_indices = [i for i, bit in enumerate(individual) if bit == 1]
        if not placed_indices:
            return (0,)

        # Create node list only for non-jitted functions (connectivity)
        temp_nodes_for_conn = []
        placed_nodes_pos_list = []  # For coverage calculation
        for idx in placed_indices:
            pos = potential_sites[idx]["pos"]
            temp_nodes_for_conn.append(
                {"id": f"GA_{idx}", "type": "SENSOR", "pos": pos}
            )  # Treat as Sensor for conn check
            placed_nodes_pos_list.append(pos)

        temp_nodes_for_conn.append(
            {"id": BASE_STATION_ID, "type": "BASE_STATION", "pos": BASE_STATION_POS}
        )

        # Call potentially JITted calculate_coverage
        # Need to pass node list structured for calculate_coverage to extract positions
        temp_nodes_for_cov = [
            {"type": "TEMP", "pos": pos} for pos in placed_nodes_pos_list
        ]
        sensor_coverage = calculate_coverage(
            temp_nodes_for_cov, target_points, SENSOR_MAX_RANGE_M, "TEMP"
        )
        camera_coverage = calculate_coverage(
            temp_nodes_for_cov, target_points, CAMERA_RANGE_M, "TEMP"
        )

        # Call non-JITted connectivity check
        connectivity_fraction, _ = check_connectivity(
            temp_nodes_for_conn, COMMUNICATION_RANGE_M, BASE_STATION_ID
        )

        # Cost
        num_placed_nodes = len(placed_indices)
        estimated_cost = num_placed_nodes * NODE_TYPES["SENSOR"]["cost"]

        # Combine scores and penalties (as before)
        fitness = (WEIGHT_SENSOR_COVERAGE * sensor_coverage) + (
            WEIGHT_CAMERA_COVERAGE * camera_coverage
        )
        if connectivity_fraction < 1.0:
            fitness -= WEIGHT_CONNECTIVITY_PENALTY * (1.0 - connectivity_fraction)
        if sensor_coverage < MIN_ACCEPTABLE_SENSOR_COVERAGE:
            fitness -= SENSOR_COVERAGE_PENALTY * (
                MIN_ACCEPTABLE_SENSOR_COVERAGE - sensor_coverage
            )
        fitness -= WEIGHT_COST * estimated_cost
        return (fitness,)

    # --- GA Setup (Keep DEAP setup as before) ---
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=num_potential_sites,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate", evaluate_deployment_ga
    )  # Register the fitness function
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- Post-Optimization Type Assignment (Keep corrected version) ---
    def assign_node_types_to_solution(best_individual_indices, potential_sites_list):
        print(
            "\nAssigning Node Types to Optimized Locations (Corrected Relay Logic)..."
        )
        if not best_individual_indices:
            print("  No locations selected by GA.")
            return []
        assigned_roles = {site_id: "SENSOR" for site_id in best_individual_indices}
        placed_nodes_info = [
            {"site_id": site_id, "pos": potential_sites_list[site_id]["pos"]}
            for site_id in best_individual_indices
        ]
        placed_nodes_map = {info["site_id"]: info for info in placed_nodes_info}

        # Step 1: Assign CAMERAS
        print("  Assigning Cameras...")
        camera_candidates = list(placed_nodes_info)
        assigned_camera_count = 0
        target_points_uncovered_by_cam = set(target_points)
        cameras_assigned_site_ids = set()
        while target_points_uncovered_by_cam and camera_candidates:
            best_candidate = None
            max_covered_count = -1
            candidate_points_covered = {}
            for candidate_info in camera_candidates:
                site_id = candidate_info["site_id"]
                if site_id not in candidate_points_covered:
                    candidate_points_covered[site_id] = get_covered_points(
                        candidate_info, target_points, CAMERA_RANGE_M
                    )  # Uses JIT helper
                newly_covered_points = candidate_points_covered[site_id].intersection(
                    target_points_uncovered_by_cam
                )
                count = len(newly_covered_points)
                if count > max_covered_count:
                    max_covered_count = count
                    best_candidate = candidate_info
            if best_candidate and max_covered_count >= 0:
                site_id = best_candidate["site_id"]
                assigned_roles[site_id] = "CAMERA"
                cameras_assigned_site_ids.add(site_id)
                assigned_camera_count += 1
                target_points_uncovered_by_cam -= candidate_points_covered[site_id]
                camera_candidates.remove(best_candidate)
            else:
                break
        print(f"    Assigned {assigned_camera_count} camera roles.")

        # Step 2: Identify Essential Sensors
        print("  Identifying essential Sensors...")
        sensor_candidates_site_ids = (
            set(best_individual_indices) - cameras_assigned_site_ids
        )
        essential_sensor_site_ids = set()
        target_points_uncovered_by_sens = set(target_points)
        sensor_coverage_map = {}
        for site_id in sensor_candidates_site_ids:
            node_info = placed_nodes_map[site_id]
            sensor_coverage_map[site_id] = get_covered_points(
                node_info, target_points, SENSOR_MAX_RANGE_M
            )  # Uses JIT helper
        while target_points_uncovered_by_sens and sensor_candidates_site_ids:
            best_sensor_site_id = -1
            max_new_coverage = -1
            for site_id in list(sensor_candidates_site_ids):
                if site_id not in sensor_coverage_map:
                    continue
                newly_covered_points = sensor_coverage_map[site_id].intersection(
                    target_points_uncovered_by_sens
                )
                count = len(newly_covered_points)
                if count > max_new_coverage:
                    max_new_coverage = count
                    best_sensor_site_id = site_id
            if best_sensor_site_id != -1 and max_new_coverage >= 0:
                essential_sensor_site_ids.add(best_sensor_site_id)
                target_points_uncovered_by_sens -= sensor_coverage_map[
                    best_sensor_site_id
                ]
                sensor_candidates_site_ids.remove(best_sensor_site_id)
                del sensor_coverage_map[best_sensor_site_id]
            else:
                break
        print(
            f"    Identified {len(essential_sensor_site_ids)} essential sensor roles."
        )
        potential_relay_site_ids = (
            set(best_individual_indices)
            - cameras_assigned_site_ids
            - essential_sensor_site_ids
        )
        print(
            f"    Identified {len(potential_relay_site_ids)} potential relay locations."
        )

        # Step 3: Assign RELAYS for Connectivity
        print("  Assigning Relays for Connectivity...")
        current_active_nodes = []
        nodes_to_connect = []
        temp_final_roles = {}
        node_id_counter = 0
        for site_id in essential_sensor_site_ids:  # Add essential sensors
            node_id = f"Opt_{node_id_counter}"
            node = {
                "id": node_id,
                "type": "SENSOR",
                "pos": placed_nodes_map[site_id]["pos"],
                "site_id": site_id,
            }
            current_active_nodes.append(node)
            nodes_to_connect.append(node)
            temp_final_roles[site_id] = "SENSOR"
            node_id_counter += 1
        for site_id in cameras_assigned_site_ids:  # Add cameras
            if site_id not in temp_final_roles:
                node_id = f"Opt_{node_id_counter}"
                node = {
                    "id": node_id,
                    "type": "CAMERA",
                    "pos": placed_nodes_map[site_id]["pos"],
                    "site_id": site_id,
                }
                current_active_nodes.append(node)
                nodes_to_connect.append(node)
                temp_final_roles[site_id] = "CAMERA"
                node_id_counter += 1
            else:
                temp_final_roles[site_id] = "CAMERA"
                [n for n in current_active_nodes if n.get("site_id") == site_id][0][
                    "type"
                ] = "CAMERA"
        current_active_nodes.append(
            {"id": BASE_STATION_ID, "type": "BASE_STATION", "pos": BASE_STATION_POS}
        )  # Add BS
        relay_site_ids_assigned = set()
        available_potential_relays = set(potential_relay_site_ids)
        iteration_count = 0
        max_iterations = len(potential_relay_site_ids) + 5
        while iteration_count < max_iterations:
            iteration_count += 1
            nodes_to_connect_ids = {n["id"] for n in nodes_to_connect}
            fraction_connected, unconnected_ids = check_connectivity(
                current_active_nodes, COMMUNICATION_RANGE_M, BASE_STATION_ID
            )
            unconnected_essential_ids = nodes_to_connect_ids.intersection(
                unconnected_ids
            )
            if not unconnected_essential_ids:
                print(f"    Connectivity established (Iteration {iteration_count}).")
                break
            # print(f"    Iter {iteration_count}: Connectivity {fraction_connected*100:.2f}% ({len(unconnected_essential_ids)} essential nodes unconnected).") # Verbose
            if not available_potential_relays:
                print(
                    f"    WARNING: No more potential relays (Iteration {iteration_count}), connectivity incomplete!"
                )
                break
            best_relay_to_add_site_id = -1
            max_newly_connected_count = -1
            for relay_site_id in available_potential_relays:
                sim_relay_node = {
                    "id": f"SimRelay_{relay_site_id}",
                    "type": "RELAY",
                    "pos": placed_nodes_map[relay_site_id]["pos"],
                    "site_id": relay_site_id,
                }
                sim_active_nodes = current_active_nodes + [sim_relay_node]
                _, sim_unconnected_ids = check_connectivity(
                    sim_active_nodes, COMMUNICATION_RANGE_M, BASE_STATION_ID
                )
                sim_unconnected_essential = nodes_to_connect_ids.intersection(
                    sim_unconnected_ids
                )
                num_now_connected = len(
                    unconnected_essential_ids - sim_unconnected_essential
                )
                if num_now_connected > max_newly_connected_count:
                    max_newly_connected_count = num_now_connected
                    best_relay_to_add_site_id = relay_site_id
            if best_relay_to_add_site_id != -1 and max_newly_connected_count > 0:
                # print(f"    Assigning site {best_relay_to_add_site_id} as RELAY (connects {max_newly_connected_count} new).") # Verbose
                relay_node = {
                    "id": f"Opt_{node_id_counter}",
                    "type": "RELAY",
                    "pos": placed_nodes_map[best_relay_to_add_site_id]["pos"],
                    "site_id": best_relay_to_add_site_id,
                }
                current_active_nodes.append(relay_node)
                relay_site_ids_assigned.add(best_relay_to_add_site_id)
                available_potential_relays.remove(best_relay_to_add_site_id)
                temp_final_roles[best_relay_to_add_site_id] = "RELAY"
                node_id_counter += 1
            else:
                print(
                    f"    WARNING: Could not find useful relay in iteration {iteration_count}. Stopping relay assignment."
                )
                break
        else:
            print(
                f"    WARNING: Reached max iterations ({max_iterations}) in relay loop."
            )
        print(f"    Assigned {len(relay_site_ids_assigned)} relay roles.")

        # Finalize Node List
        final_nodes = []
        node_id_counter = 0
        final_site_ids = set(temp_final_roles.keys())
        for site_id in sorted(list(final_site_ids)):
            role = temp_final_roles[site_id]
            node = {
                "id": f"Opt_{node_id_counter}",
                "type": role,
                "pos": placed_nodes_map[site_id]["pos"],
            }
            final_nodes.append(node)
            node_id_counter += 1
        final_nodes.append(
            {"id": BASE_STATION_ID, "type": "BASE_STATION", "pos": BASE_STATION_POS}
        )
        print(f"Type Assignment Complete. Final count: {len(final_nodes)-1}")
        return final_nodes

    def assign_node_types_to_solution(best_individual_indices, potential_sites_list):
        print("\nAssigning Node Types to Optimized Locations (Prioritize Relay for Conn)...")
        if not best_individual_indices: print("  No locations selected by GA."); return []

        # Initial Data Structures
        final_roles = {} # site_id -> assigned role ('CAMERA', 'SENSOR', 'RELAY')
        placed_nodes_info = [{'site_id': site_id, 'pos': potential_sites_list[site_id]['pos']} for site_id in best_individual_indices]
        placed_nodes_map = {info['site_id']: info for info in placed_nodes_info}

        # Use global target_points defined earlier
        global target_points
        if not target_points: # Ensure target_points is populated
            print("Error: target_points not defined globally for assignment.")
            return [] # Or repopulate here

        # --- Step 1: Assign CAMERAS greedily ---
        print("  Step 1: Assigning Cameras...")
        camera_candidates = list(placed_nodes_info); assigned_camera_count = 0
        target_points_uncovered_by_cam = set(target_points); cameras_assigned_site_ids = set()
        # (Camera assignment logic remains the same as the previous 'corrected' version)
        while target_points_uncovered_by_cam and camera_candidates:
            best_candidate = None; max_covered_count = -1; candidate_points_covered = {}
            for candidate_info in camera_candidates:
                site_id = candidate_info['site_id']
                if site_id not in candidate_points_covered: candidate_points_covered[site_id] = get_covered_points(candidate_info, target_points, CAMERA_RANGE_M)
                newly_covered_points = candidate_points_covered[site_id].intersection(target_points_uncovered_by_cam)
                count = len(newly_covered_points)
                if count > max_covered_count: max_covered_count = count; best_candidate = candidate_info
            if best_candidate and max_covered_count >= 0:
                site_id = best_candidate['site_id']; final_roles[site_id] = 'CAMERA'; cameras_assigned_site_ids.add(site_id); assigned_camera_count += 1
                # Only remove points covered by the chosen camera from the *camera* coverage check set
                target_points_uncovered_by_cam -= candidate_points_covered[site_id]
                camera_candidates.remove(best_candidate)
            else: break
        print(f"    Assigned {assigned_camera_count} camera roles.")

        # --- Step 2: Iteratively Assign Sensors & Relays ---
        print("  Step 2: Assigning Sensors & Relays for Coverage & Connectivity...")

        # Nodes selected by GA but not yet assigned (potential Sensors/Relays)
        remaining_candidate_ids = set(best_individual_indices) - cameras_assigned_site_ids

        # Nodes currently active and assigned (start with BS and Cameras)
        current_active_nodes = [{"id": BASE_STATION_ID, "type": "BASE_STATION", "pos": BASE_STATION_POS}]
        essential_nodes_to_connect = [] # Sensors + Cameras
        for site_id in cameras_assigned_site_ids:
            node_id = f"Opt_Cam_{site_id}"
            node = {"id": node_id, "type": "CAMERA", "pos": placed_nodes_map[site_id]['pos'], "site_id": site_id}
            current_active_nodes.append(node)
            essential_nodes_to_connect.append(node) # Cameras need connection

        target_points_uncovered_by_sens = set(target_points)
        # Calculate initial sensor coverage provided *only* by cameras (if any cameras can sense)
        # Assuming cameras DON'T provide sensor coverage for this problem. If they did, adjust here.

        sensors_assigned_count = 0
        relays_assigned_count = 0

        iteration = 0
        max_iterations = len(remaining_candidate_ids) + 5 # Safety break

        while iteration < max_iterations:
            iteration += 1

            # --- Check Goals ---
            # Calculate current sensor coverage from assigned Sensors
            current_sensors = [n for n in current_active_nodes if n['type'] == 'SENSOR']
            current_sensor_coverage = calculate_coverage(current_sensors, target_points, SENSOR_MAX_RANGE_M, "SENSOR")

            # Check connectivity of essential nodes
            essential_node_ids = {n['id'] for n in essential_nodes_to_connect}
            _, unconnected_ids = check_connectivity(current_active_nodes, COMMUNICATION_RANGE_M, BASE_STATION_ID)
            unconnected_essential_node_ids = essential_node_ids.intersection(unconnected_ids)

            # --- Termination Conditions ---
            sensor_coverage_met = current_sensor_coverage >= MIN_ACCEPTABLE_SENSOR_COVERAGE
            connectivity_met = not unconnected_essential_node_ids

            print(f"    Iter {iteration}: Candidates={len(remaining_candidate_ids)}, SensorCov={current_sensor_coverage*100:.2f}%, ConnOK={connectivity_met}, NeedToConnect={len(unconnected_essential_node_ids)}")

            if sensor_coverage_met and connectivity_met:
                print("    Goals met: Sensor coverage and connectivity established.")
                break
            if not remaining_candidate_ids:
                print("    No more candidates available.")
                if not sensor_coverage_met: print("    WARNING: Sensor coverage goal NOT met.")
                if not connectivity_met: print("    WARNING: Connectivity goal NOT met.")
                break

            # --- Find Best Candidate to Add ---
            best_candidate_id = -1
            best_score = -float('inf') # Score combines connectivity benefit and coverage benefit
            best_role_for_candidate = None # 'RELAY' or 'SENSOR'

            # Prioritize connection, then coverage
            connectivity_priority_weight = 1000
            coverage_priority_weight = 1

            for candidate_site_id in remaining_candidate_ids:
                candidate_info = placed_nodes_map[candidate_site_id]

                # --- Evaluate Connectivity Benefit ---
                # Simulate adding this node as a RELAY
                sim_relay_node = {"id": f"Sim_{candidate_site_id}", "type": "RELAY", "pos": candidate_info['pos'], "site_id": candidate_site_id}
                sim_nodes_conn = current_active_nodes + [sim_relay_node]
                _, sim_unconnected_ids_conn = check_connectivity(sim_nodes_conn, COMMUNICATION_RANGE_M, BASE_STATION_ID)
                sim_unconnected_essential_conn = essential_node_ids.intersection(sim_unconnected_ids_conn)
                # Benefit = how many previously unconnected essential nodes are now connected
                connectivity_benefit = len(unconnected_essential_node_ids - sim_unconnected_essential_conn)

                # --- Evaluate Coverage Benefit ---
                # Simulate adding this node as a SENSOR
                points_covered_by_candidate = get_covered_points(candidate_info, target_points, SENSOR_MAX_RANGE_M)
                # Benefit = how many previously uncovered target points are now covered
                coverage_benefit = len(points_covered_by_candidate.intersection(target_points_uncovered_by_sens))

                # --- Combine Score ---
                # If connectivity is not yet met, prioritize nodes that help connect
                current_score = 0
                role_if_chosen = None

                if not connectivity_met and connectivity_benefit > 0:
                    # If it primarily helps connectivity, consider it a relay
                    current_score = connectivity_benefit * connectivity_priority_weight + coverage_benefit * coverage_priority_weight
                    role_if_chosen = 'RELAY' # Tentative role if chosen based on this score
                elif not sensor_coverage_met and coverage_benefit > 0:
                    # If connectivity is met OR this node doesn't help connect, but sensor cov is not met and this node helps cover
                    current_score = connectivity_benefit * connectivity_priority_weight + coverage_benefit * coverage_priority_weight
                    role_if_chosen = 'SENSOR' # Tentative role
                else:
                    # If both goals met OR this node provides no benefit to unmet goals
                    # Could add based on secondary criteria (e.g., robustness, lowest cost?) - skip for now
                    current_score = -1 # Don't select if no primary benefit
                    role_if_chosen = None

                # --- Update Best Candidate ---
                if current_score > best_score:
                    best_score = current_score
                    best_candidate_id = candidate_site_id
                    best_role_for_candidate = role_if_chosen

            # --- Add the Chosen Candidate ---
            if best_candidate_id != -1 and best_role_for_candidate is not None:
                chosen_node_info = placed_nodes_map[best_candidate_id]
                print(f"    Adding site {best_candidate_id} as {best_role_for_candidate} (Score: {best_score:.1f})")

                final_roles[best_candidate_id] = best_role_for_candidate

                # Add to active nodes and update essential lists/coverage sets
                new_node_id = f"Opt_{best_role_for_candidate}_{best_candidate_id}"
                new_node = {"id": new_node_id, "type": best_role_for_candidate, "pos": chosen_node_info['pos'], "site_id": best_candidate_id}
                current_active_nodes.append(new_node)

                if best_role_for_candidate == 'SENSOR':
                    essential_nodes_to_connect.append(new_node) # Sensors need connection
                    # Update uncovered sensor points using cached/recalculated coverage
                    points_covered = get_covered_points(chosen_node_info, target_points, SENSOR_MAX_RANGE_M)
                    target_points_uncovered_by_sens -= points_covered
                    sensors_assigned_count += 1
                elif best_role_for_candidate == 'RELAY':
                    relays_assigned_count +=1

                remaining_candidate_ids.remove(best_candidate_id) # Remove added node

            else:
                print("    Could not find a beneficial node to add in this iteration. Stopping.")
                if not sensor_coverage_met: print("    WARNING: Sensor coverage goal likely NOT met.")
                if not connectivity_met: print("    WARNING: Connectivity goal likely NOT met.")
                break # No progress

        else: # Safety break
            print(f"    WARNING: Reached max iterations ({max_iterations}).")
            # Final checks after loop exit
            current_sensors = [n for n in current_active_nodes if n['type'] == 'SENSOR']
            current_sensor_coverage = calculate_coverage(current_sensors, target_points, SENSOR_MAX_RANGE_M, "SENSOR")
            essential_node_ids = {n['id'] for n in essential_nodes_to_connect}
            _, unconnected_ids = check_connectivity(current_active_nodes, COMMUNICATION_RANGE_M, BASE_STATION_ID)
            unconnected_essential_node_ids = essential_node_ids.intersection(unconnected_ids)
            if current_sensor_coverage < MIN_ACCEPTABLE_SENSOR_COVERAGE : print("    WARNING: Final sensor coverage goal NOT met.")
            if unconnected_essential_node_ids : print("    WARNING: Final connectivity goal NOT met.")

        print(f"    Assigned {sensors_assigned_count} sensor roles.")
        print(f"    Assigned {relays_assigned_count} relay roles.")

        # --- Finalize Node List ---
        final_nodes = []
        node_id_counter = 0
        # Include only nodes that were actually assigned a role in final_roles
        for site_id in sorted(list(final_roles.keys())):
            role = final_roles[site_id]
            node = {"id": f"Opt_{node_id_counter}", "type": role, "pos": placed_nodes_map[site_id]['pos']}
            final_nodes.append(node)
            node_id_counter += 1

        final_nodes.append({"id": BASE_STATION_ID, "type": "BASE_STATION", "pos": BASE_STATION_POS})
        print(f"Type Assignment Complete. Final count: {len(final_nodes)-1}")
        return final_nodes

    # --- Run GA ---
    # **** IMPORTANT: Use the run_ga_optimization_parallel() function from the previous answer ****
    # **** Make sure it calls the new assign_node_types_to_solution defined above ****
    def run_ga_optimization_parallel(): # Renamed function
        if not DEAP_AVAILABLE: print("Cannot run optimization: DEAP library not available."); return None
        try:
            num_workers = multiprocessing.cpu_count()
            print(f"Setting up multiprocessing pool with {num_workers} workers.")
            pool = multiprocessing.Pool(processes=num_workers)
            toolbox.register("map", pool.map)
        except NotImplementedError:
            print("Could not determine CPU count. Running GA sequentially.")
            toolbox.unregister("map"); pool = None

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean); stats.register("std", np.std); stats.register("min", np.min); stats.register("max", np.max)
        print("\n--- Starting Genetic Algorithm Optimization (Parallel) ---"); start_time = time.time()
        final_pop = None; best_ind = None
        try:
            # Need population initialized here if not passed
            global pop # If pop is defined globally or passed in
            if 'pop' not in globals(): pop = toolbox.population(n=POPULATION_SIZE)

            final_pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB, ngen=NUM_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
            if hof: best_ind = hof[0]
        except Exception as e:
            print(f"\n*** Error during GA execution: {e} ***"); import traceback; traceback.print_exc(); return None
        finally:
            if pool is not None: print("Closing multiprocessing pool."); pool.close(); pool.join()

        end_time = time.time(); print(f"--- Optimization Finished (Duration: {end_time - start_time:.2f} seconds) ---")
        if not best_ind:
            if hof and len(hof) > 0: best_ind = hof[0]
            else: print("Error: HallOfFame is empty or GA failed."); return None
        print(f"\nBest Individual Fitness: {best_ind.fitness.values[0]:.4f}")
        placed_indices = [i for i, bit in enumerate(best_ind) if bit == 1]
        print(f"GA selected {len(placed_indices)} node locations.")

        # Calls the NEW assignment function
        optimized_nodes_with_types = assign_node_types_to_solution(placed_indices, potential_sites)
        return optimized_nodes_with_types

# --- Main Execution (Keep as before, ensure pop initialized for GA call) ---
if __name__ == "__main__":
    print("="*40); print(" Method 1: Heuristic Placement"); print("="*40)
    target_points = []
    for x in range(COVERAGE_TARGET_STEP // 2, AREA_WIDTH_M, COVERAGE_TARGET_STEP):
        for y in range(COVERAGE_TARGET_STEP // 2, AREA_HEIGHT_M, COVERAGE_TARGET_STEP):
            target_points.append((x, y))
    target_points_global_arr = np.array(target_points, dtype=np.float64) if NUMBA_AVAILABLE else None

    heuristic_nodes = place_heuristic_nodes()
    analyze_deployment(heuristic_nodes, COMMUNICATION_RANGE_M, SENSOR_MAX_RANGE_M, CAMERA_RANGE_M, target_points, BASE_STATION_ID)
    plot_deployment(heuristic_nodes, AREA_WIDTH_M, AREA_HEIGHT_M, PLOT_SIZE_M, title="Heuristic WSN Deployment")

    print("\n\n"+"="*40); print(" Method 2: Genetic Algorithm Optimization"); print("="*40)
    if DEAP_AVAILABLE:
        # Initialize population globally or ensure toolbox is ready
        if 'individual' in toolbox.__dict__:
             pop = toolbox.population(n=POPULATION_SIZE) # Initialize population for the run
             optimized_nodes = run_ga_optimization_parallel()
             if optimized_nodes:
                 print("\n--- Analysis of Optimized Deployment ---")
                 analyze_deployment(optimized_nodes, COMMUNICATION_RANGE_M, SENSOR_MAX_RANGE_M, CAMERA_RANGE_M, target_points, BASE_STATION_ID)
                 plot_deployment(optimized_nodes, AREA_WIDTH_M, AREA_HEIGHT_M, PLOT_SIZE_M, title="Optimized WSN Deployment (GA - Prioritize Relay)")
             else: print("Optimization failed or produced no result.")
        else: print("Error: DEAP Toolbox not fully initialized before GA run.")
    else: print("Skipping GA optimization as DEAP library is not installed.")

    print("\nDone.")