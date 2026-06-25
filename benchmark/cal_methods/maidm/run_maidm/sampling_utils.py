import os
import random
import pandas as pd
from pickle import UnpicklingError
import xml.etree.ElementTree as ET


def load_trace(cache):
    """
    Load the trace from a pickle file. If the file is broken, it will be removed.
    """
    if os.path.exists(cache):
        try:
            tr = pd.read_pickle(cache)
            print("Load trace", cache, ": done!")
        except UnpicklingError:
            os.remove(cache)
            print("Removed broken cache:", cache)
    else:
        print("No file found in your path!")

    return tr


def update_routes(
    rou_xml_file_path,
    tr,
    num_vtypes_to_insert,
    vtype_id_prefix,
    vtype_dist_id,
    car_follow_model="IDM",
    rand_seed=1,
):
    """
    Function to sample vehicle types from the posterior distribution and update the SUMO route
    XML file with a vehicle distribution comprised of these types.

    Parameters:
        rou_xml_file_path (str): Path to the SUMO route XML file to be updated.
        tr: Trace containing the posterior distribution of vehicle parameters.
        num_vtypes_to_insert (int): Number of vehicle types to insert into the XML file.
        vtype_id_prefix (str): Prefix for the vehicle type IDs.
        vtype_dist_id (str): ID for the vehicle type distribution element.
        car_follow_model (str): SUMO car-following model to be used for the vehicle types.
        rand_seed (int): Random seed for reproducibility.

    Returns:
        None: The function modifies the XML file in place.
    """
    if not os.path.exists(rou_xml_file_path):
        raise FileNotFoundError(
            f"The specified XML file does not exist: {rou_xml_file_path}"
        )

    try:
        # Parse the XML file
        tree = ET.parse(rou_xml_file_path)
        root = tree.getroot()

        # Clear existing vType and vTypeDistribution elements
        for element in list(root): # Iterate over a copy to allow modification during iteration
            if element.tag in ["vType", "vTypeDistribution"]:
                root.remove(element)

        # Find the index of the first <route> element
        first_route_index = -1
        for i, element in enumerate(root):
            if element.tag == "route":
                first_route_index = i
                break

        if first_route_index == -1:
            print(
                "No <route> elements found in the XML. Appending <vType> elements to the end."
            )
            first_route_index = len(root)  # Append to the end if no route found

        # Set random seed
        random.seed(rand_seed)

        # Get the number of draws in the posterior for later sampling
        num_draws_in_posterior = tr.posterior.mu.mean(dim="chain").sizes["draw"]

        if num_vtypes_to_insert > 0:
            # Create the vTypeDistribution element first
            vtype_dist_element = ET.Element("vTypeDistribution", id=vtype_dist_id)
            equal_probability = 1.0 / num_vtypes_to_insert

            # Insert the specified number of <vType> elements
            for i in range(num_vtypes_to_insert):
                # Generate unique ID for each vType within the distribution
                vtype_id = f"{vtype_id_prefix}_{i+1}"

                # Sample parameters from the posterior distribution
                random_draw_index = random.randint(0, num_draws_in_posterior - 1)
                sampled_params = tr.posterior.mu.mean(dim="chain").isel(
                    draw=random_draw_index
                )

                # Create the <vType> element as a sub-element of vTypeDistribution
                vtype_element = ET.SubElement(
                    vtype_dist_element,
                    "vType",
                    id=vtype_id,
                    carFollowModel=car_follow_model,
                    minGap=str(sampled_params.values[1]),  # m
                    desiredMaxSpeed=str(sampled_params.values[0]),  # m/s
                    accel=str(sampled_params.values[3]),  # m/s^2
                    decel=str(sampled_params.values[4]),  # m/s^2
                    tau=str(sampled_params.values[2]),  # s
                    probability=str(equal_probability),
                )

            # Insert the complete vTypeDistribution element into the root
            root.insert(first_route_index, vtype_dist_element)
            print(f"Created vTypeDistribution '{vtype_dist_id}' with {num_vtypes_to_insert} vTypes defined inline.")
        else:
            print("num_vtypes_to_insert is 0, so no vTypeDistribution was created.")

        ET.indent(root, space="    ", level=0)  # Use 4 spaces for indentation
        tree.write(rou_xml_file_path, encoding="utf-8", xml_declaration=True)
        print(
            f"Successfully updated routes file w/ {num_vtypes_to_insert} vehicle types in {vtype_dist_id} vtype distribution and saved to '{rou_xml_file_path}'"
        )

    except ET.ParseError as e:
        print(f"Error parsing the XML file: {e}")
    except Exception as e:
        print(f"An error occurred while processing the XML file: {e}")


def update_flows(flows_xml_file_path, vtype_dist_id):
    """
    Function to update the SUMO flow XML file with a specified vehicle type distribution.
    Parameters:
        flows_xml_file_path (str): Path to the SUMO flow XML file to be updated.
        vtype_dist_id (str): ID for the vehicle type distribution to be set in each flow.
    Returns:
        None: The function modifies the XML file in place.
    """
    try:
        # Parse the XML file
        tree = ET.parse(flows_xml_file_path)
        root = tree.getroot()

        # Change the root element tag from 'additional' to 'routes'
        # (will prevent a SUMO warning)
        root.tag = "routes"

        # Clear existing vType elements if they exist in the file
        for vtype in root.findall("type"):
            root.remove(vtype)

        # Add the type element to each flow
        for flow_element in root.findall("flow"):
            flow_element.set("type", vtype_dist_id)

        tree.write(flows_xml_file_path, encoding="utf-8", xml_declaration=True)
        print(
            f"Successfully updated flows w/ vehicle types and saved to '{flows_xml_file_path}'"
        )

    except FileNotFoundError:
        print(f"Error: The file '{flows_xml_file_path}' was not found.")
    except ET.ParseError as e:
        print(f"Error parsing XML from '{flows_xml_file_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def update_vtype_colors(xml_file_path):
    """
    Reads a SUMO route XML file, updates each vType element to have a
    realistic color, and writes the modified XML back to the same file.

    Args:
        xml_file_path (str): The path to the SUMO route XML file (e.g., 'maidm.rou.xml').
    """
    try:
        car_colors = [
            "240,240,240",  # Off-White / Light Grey
            "40,40,40",     # Dark Grey / Charcoal
            "180,180,180",  # Silver / Medium Grey
            "25,25,112",    # Midnight Blue / Dark Navy
            "128,0,0",      # Maroon / Dark Red
            "30,70,30",    # Forest Green / Dark Green
        ]

        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Set seed
        random.seed(1)  # For reproducibility

        # Find all vType elements within vTypeDistribution
        vtype_distribution_found = False
        for vtd in root.findall(".//vTypeDistribution"):
            vtype_distribution_found = True
            vtypes = vtd.findall("vType")

            if not vtypes:
                print(f"No vType elements found within vTypeDistribution '{vtd.get('id')}'.")
                continue

            # Assign a random realistic color to each vType
            for i, vtype in enumerate(vtypes):
                # Get a random color from the list
                color_to_assign = random.choice(car_colors)
                vtype.set("color", color_to_assign)

        if not vtype_distribution_found:
            print("No <vTypeDistribution> element found in the XML file.")
            # As a fallback, check for vType elements directly under <routes>
            print("Checking for <vType> elements directly under <routes>...")
            vtypes_direct = root.findall("vType")
            if vtypes_direct:
                for i, vtype in enumerate(vtypes_direct):
                    color_to_assign = random.choice(car_colors)
                    vtype.set("color", color_to_assign)
            else:
                print("No <vType> elements found directly under <routes> either.")

        # Write the modified XML back to the file
        tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)
        print(f"Successfully vehicle colors in '{xml_file_path}' to be more realistic.")

    except FileNotFoundError:
        print(f"Error: The file '{xml_file_path}' was not found.")
    except ET.ParseError as e:
        print(f"Error parsing XML file '{xml_file_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")