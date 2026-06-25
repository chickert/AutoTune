"""
File that generates the detector dictionary & associated detector XML files for the SUMO simulation based on the I-24 SUMO network and 
provided vehicle/speed detector lat/long coordinates (such as RDS or inductor loop detectors).
This adds lane-level detectors at each lane on the netwwork edge closest to the detector's lat/long position.
For visual inspection, it also generates POI files for the original RDS detector GPS coordinates and the lane-level mappings of the detectors.
"""

import pandas as pd
import sumolib
import xml.etree.ElementTree as ET
import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]


### Parameters ###

sim_files_dir = str(BENCHMARK_DIR / "sim_files") + "/"                 # Path to the directory containing the SUMO simulation files
detector_data_dir = str(BENCHMARK_DIR / "build_data") + "/"            # Path to the directory containing the detector data files

# Params for map_detectors_to_sumo_network()
network_file = sim_files_dir + "i24_modded_and_trimmed_CH.net.xml"      # Path to the SUMO network file
detector_source_file = detector_data_dir + "rds_detectors_info.csv"     # Path to the file containing detector information
i24_milemarkers = (58.7, 62.9)                                          # Tuple containing the milemarkers of the I-24 segment of interest
search_radius = 100                                                     # The radius around a detector lat/long point in which to search for network edges
onramp_edge_ids_to_exclude = []                                         # List of edge IDs to exclude from consideration when mapping detectors to the network
milemarker_col_title = "tdot_milemarker"                                # The title of the column in the detector_source_file that contains a detector's milemarker location
detectors_to_exclude = ['555-eastbound']                                # List of detector IDs to exclude from the returned dictionary (usually because of a known issue with I-24 data at that milemarker/direction)               
verbose = False                                                         # Whether to print detailed information about the mapping process
detector_dict_path = detector_data_dir + "sumo_net_detector_dict.json"  # Path for where to save the detector dictionary

# Params for generate_detector_xml()
output_file = sim_files_dir + "04-20_detectors.xml"                     # Path to the output file
detector_type = "inductionLoop"                                         # Type of detector to generate (e.g., "inductionLoop")
freq = "30"                                                             # Frequency of the detector (seconds)

# Params for generate_detector_POI_orig_xml()
detector_orig_gps_file = sim_files_dir + "04-20_gps.xml"                # Path to the output file for the original GPS coordinates of the detectors
detector_lanelocs_poi_file = sim_files_dir + "04-20_lanelocs.xml"       # Path to the output file for the lane-level locations of the detectors
orig_detectors_img_file = sim_files_dir + "square_orange.png"           # Image file for the original GPS coordinates of the detectors
lanelocs_detectors_img_file = sim_files_dir + "square_gold.png"         # Image file for the lane-level locations of the detectors
width = 10                                                              # Width of the image for the original GPS coordinates of the detectors
height = 10                                                             # Height of the image for the original GPS coordinates of the detectors

# Params for generate_detector_POI_laneloc_xml()
width = 2.5                                                             # Width of the image for the lane-level locations of the detectors
height = 2.5                                                            # Height of the image for the lane-level locations of the detectors

##################


def map_detectors_to_sumo_network(
    network_file,
    detector_source_file,
    i24_milemarkers,
    search_radius,
    onramp_edge_ids_to_exclude,
    milemarker_col_title,
    detectors_to_exclude=[],
    verbose=False,
    detector_dict_path=None,
):
    """
    Maps detectors to the SUMO network based on their lat/long positions and the nearest edge going the correct direction.

    Args:
        network_file (str): Path to the SUMO network file.
        detector_source_file (str): Path to the file containing detector information.
        i24_milemarkers (tuple): Tuple containing the milemarkers of the I-24 segment of interest.
        search_radius (int): The radius around a detector lat/long point in which to search for network edges.
        onramp_edge_ids_to_exclude (list): List of edge IDs to exclude from consideration when mapping detectors to the network.
            (Useful for excluding onramps that are closer to a detector than the desired highway edge.)
        milemarker_col_title (str): The title of the column in the detector_source_file that contains a detector's milemarker location.
        detectors_to_exclude (list): List of detector IDs to exclude from the returned dictionary (usually because of a known issue with I-24 data at that milemarker/direction).
        verbose (bool): Whether to print detailed information about the mapping process.
        detector_dict_path (str): Path to save the detector dictionary output file.

    Returns:
        dict: A dictionary that can be used to build a SUMO detectors .xml file (edge ID, # lanes on that edge, detector's position along the edge, original xy coords for checking, and milemarker).
    """

    sumo_net_detector_dict = {}

    # Load network file for gps-xy conversion and edge/lane information
    net = sumolib.net.readNet(network_file)

    # Load file with detector information
    detectors_df = pd.read_csv(detector_source_file)

    num_skipped_detectors = 0
    num_processed_detectors = 0
    num_lane_level_detectors = 0

    # Iterate over each detector in input file
    for i, row in detectors_df.iterrows():
        # Filter out detectors outside I-24 milemarkers
        if (
            row["tdot_milemarker"] < i24_milemarkers[0]
            or row["tdot_milemarker"] > i24_milemarkers[1]
        ):
            if verbose:
                print(
                    f"{i}: Skipping detector {row['detector_id']} at MM {row['tdot_milemarker']} because it is outside I-24 milemarkers."
                )
            num_skipped_detectors += 1
            continue

        # Extract relevant detector information
        detector_id = row["detector_id"]
        detector_road_side = row["road_side"]
        detector_lat = row["latitude"]
        detector_lon = row["longitude"]
        detector_milemarker = row[milemarker_col_title]

        if verbose:
            print(
                f"{i}: Processing detector {detector_id}-{detector_road_side} at MM {row['tdot_milemarker']}..."
            )
        num_processed_detectors += 1

        # Get the xy position of the detector
        xy_pos = net.convertLonLat2XY(detector_lon, detector_lat)

        # Returns a list of edges that are within the given radius of the given position, along with their distance from the position
        edges = net.getNeighboringEdges(xy_pos[0], xy_pos[1], search_radius)
        assert (
            len(edges) > 0
        ), f"\tCould not find any edges within specified search_radius of {search_radius}m for detector {detector_id}-{detector_road_side}."

        # Get the nearest edge that is also going the correct direction
        # Iterate over edges, starting with the closest
        nearest_edge = None
        for edge, dist_to_edge in sorted(edges, key=lambda x: x[1]):

            # Exclude edges that are not part of the main I-24 road
            if edge.getID() in onramp_edge_ids_to_exclude:
                continue

            # Determine the direction of the edge
            edge_xcoord_delta = (
                edge.getShape()[-1][0] - edge.getShape()[0][0]
            )  # x-coordinate of endpoint minus x-coordinate of start point
            # Check that the edge has a length and is not vertical
            assert (
                edge_xcoord_delta != 0
            ), f"\tEdge {edge.getID()} has no length or is vertical."
            # Extract direction
            edge_direction = "eastbound" if edge_xcoord_delta > 0 else "westbound"

            # If the edge direction is correct, then we have found the nearest edge going the correct direction
            # (Since we are iterating over edges in order of distance, this will be the nearest edge going the correct direction)
            if (edge_direction == "eastbound" and detector_road_side == "e") or (
                edge_direction == "westbound" and detector_road_side == "w"
            ):
                nearest_edge = edge
                break

        assert (
            nearest_edge is not None
        ), f"\tCould not find any edges in specified radius going the correct direction for detector {detector_id}-{detector_road_side}."

        # Get the offset position along the edge
        offset_pos_along_edge, residual_dist = (
            sumolib.geomhelper.polygonOffsetAndDistanceToPoint(
                (xy_pos[0], xy_pos[1]), nearest_edge.getShape()
            )
        )
        # Extract the number of lanes on the edge
        num_lanes = len(nearest_edge.getLanes())

        if verbose:
            print(f"\tnearest_edge: {nearest_edge.getID()}")
            print(f"\toffset: {offset_pos_along_edge}")
            print(f"\tdist (should be same as dist above): {residual_dist}")
            print(f"\tnearest_edge has {num_lanes} lanes")

        # Save the detector information
        sumo_detector_id = f"{detector_id}-{edge_direction}"

        # Ignore specified detector(s)
        if sumo_detector_id in detectors_to_exclude:
            print(f"\nOmitted detector {sumo_detector_id}, as specified by user.")
            num_skipped_detectors += 1
            num_processed_detectors -= 1    # Correct the count of processed detectors
            continue

        assert (
            sumo_detector_id not in sumo_net_detector_dict
        ), f"\tDetector {sumo_detector_id} already exists in dictionary."
        sumo_net_detector_dict[sumo_detector_id] = {
            "edge_ID": nearest_edge.getID(),
            "num_lanes": num_lanes,
            "pos": round(offset_pos_along_edge, 2),
            "xy_pos": xy_pos,
            "milemarker": detector_milemarker,
        }

        num_lane_level_detectors += num_lanes

    print(
        f"\nProcessed {num_processed_detectors} detectors and assigned to {num_lane_level_detectors} lanes."
    )
    print(
        f"Skipped {num_skipped_detectors} detectors that were outside I-24 milemarkers.\n"
    )

    if detector_dict_path:
        # Save sumo_net_detector_dict to a file for later use
        with open(detector_dict_path, "w") as f:
            json.dump(sumo_net_detector_dict, f, indent=4)
            print(f"\nSaved sumo_net_detector_dict to save_path: {detector_dict_path}\n")

    return sumo_net_detector_dict


def generate_detector_xml(
    sumo_net_detector_dict,
    output_file,
    detector_type="inductionLoop",
    freq="30",
    file="./detector_files/out.xml",
    friendlyPos="False",
):
    """
    Generates an XML file with the specified number of lane-level detectors at the appropriate positions.
    Args:
        sumo_net_detector_dict: Dictionary w/ edge ID, # lanes on that edge, detector's position along the edge
        output_file: Path to the output file
        freq: (str) Frequency of the detector (seconds)
        file: (str) Output file for the detector
        friendlyPos: (str) Whether to use friendly position
    """

    # Create root element with namespaces and schema location
    root = ET.Element(
        "additional",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
        },
    )

    # Add a detector to each lane on the given edge at the appropriate position
    num_detectors = 0
    for detector_id, detector_info in sumo_net_detector_dict.items():
        for lane_idx in range(detector_info["num_lanes"]):
            detector = ET.SubElement(root, detector_type)
            detector.set("id", f"{detector_id}_{lane_idx}")
            detector.set("lane", f"{detector_info['edge_ID']}_{lane_idx}")
            detector.set("pos", str(detector_info["pos"]))
            detector.set("freq", freq)
            detector.set("file", file)
            detector.set("friendlyPos", friendlyPos)
            num_detectors += 1

    # Create ElementTree
    tree = ET.ElementTree(root)
    # Add for pretty formatting
    ET.indent(tree, "  ")
    # Write to file
    tree.write(output_file)

    print(f"Done writing {num_detectors} detectors to {output_file}.")


def generate_detector_POI_orig_xml(
    sumo_net_detector_dict, output_file, img_file, width, height
):
    """
    Generates an XML file with the GPS coordinates of the detectors translated to XY coordinates in the SUMO map for visualization as POIs.
    Args:
        sumo_net_detector_dict: Dictionary w/ edge ID, # lanes on that edge, and XY coordinates of the detector (mapped from GPS)
        output_file: Path to the output file
        freq: Frequency of the detector
        file: Output file for the detector
        friendlyPos: Whether to use friendly position
    """

    # Create root element with namespaces and schema location
    root = ET.Element(
        "additional",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
        },
    )

    # Add a detector to each lane on the given edge at the appropriate position
    num_detectors = 0
    for detector_id, detector_info in sumo_net_detector_dict.items():
        detector = ET.SubElement(root, "poi")
        detector_x_pos, detector_y_pos = detector_info["xy_pos"]
        detector.set("id", f"{detector_id}")
        detector.set("x", f"{detector_x_pos}")
        detector.set("y", f"{detector_y_pos}")
        detector.set("type", "detector_true_loc")
        detector.set("imgFile", img_file)
        detector.set("width", f"{width}")
        detector.set("height", f"{height}")
        num_detectors += 1

    # Create ElementTree
    tree = ET.ElementTree(root)
    # Add for pretty formatting
    ET.indent(tree, "  ")
    # Write to file
    tree.write(output_file)

    print(f"Done writing {num_detectors} detectors to {output_file}.")


def generate_detector_POI_laneloc_xml(
    sumo_net_detector_dict, output_file, img_file, width, height
):
    """
    Generates an XML file with the detectors in the positions to which they are mapped in the SUMO network for visualization as POIs.
    Args:
        sumo_net_detector_dict: Dictionary w/ edge ID, # lanes on that edge, detector's position along the edge
        output_file: Path to the output file
        freq: Frequency of the detector
        file: Output file for the detector
        friendlyPos: Whether to use friendly position
    """

    # Create root element with namespaces and schema location
    root = ET.Element(
        "additional",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
        },
    )

    # Add a detector to each lane on the given edge at the appropriate position
    num_detectors = 0
    for detector_id, detector_info in sumo_net_detector_dict.items():
        for lane_idx in range(detector_info["num_lanes"]):
            detector = ET.SubElement(root, "poi")
            detector.set("id", f"{detector_id}_{lane_idx}")
            detector.set("lane", f"{detector_info['edge_ID']}_{lane_idx}")
            detector.set("pos", str(detector_info["pos"]))
            detector.set("type", "detector_inferred_lane_loc")
            detector.set("imgFile", img_file)
            detector.set("width", f"{width}")
            detector.set("height", f"{height}")
            num_detectors += 1

    # Create ElementTree
    tree = ET.ElementTree(root)
    # Add for pretty formatting
    ET.indent(tree, "  ")
    # Write to file
    tree.write(output_file)

    print(f"Done writing {num_detectors} lane-level detectors to {output_file}.")


if __name__ == "__main__":

    sumo_net_detector_dict = map_detectors_to_sumo_network(
        network_file,
        detector_source_file,
        i24_milemarkers,
        search_radius,
        onramp_edge_ids_to_exclude,
        milemarker_col_title,
        detectors_to_exclude=detectors_to_exclude,
        verbose=verbose,
        detector_dict_path=detector_dict_path,
    )

    generate_detector_xml(
        sumo_net_detector_dict, output_file, detector_type=detector_type, freq=freq
    )

    generate_detector_POI_orig_xml(
        sumo_net_detector_dict,
        detector_orig_gps_file,
        orig_detectors_img_file,
        width,
        height,
    )

    generate_detector_POI_laneloc_xml(
        sumo_net_detector_dict,
        detector_lanelocs_poi_file,
        lanelocs_detectors_img_file,
        width,
        height,
    )
