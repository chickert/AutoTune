import xml.etree.ElementTree as ET

def color_vehs(rou_file_path, vtype_dist_id):
    """
    Reads a SUMO .rou.xml file, appends a vTypeDistribution with one vType for each color,
    and writes the modified XML to a new file.
    Parameters:
        rou_file_path (str): Path to the original .rou.xml file.
        vtype_dist_id (str): ID for the vTypeDistribution to be created.
    Returns:
        None: The function modifies the XML file in place.
    """
    try:
        # Parse the existing XML file
        tree = ET.parse(rou_file_path)
        root = tree.getroot() # This should be the <routes> tag

        # Clear existing vType and vTypeDistribution elements
        for element in list(root): # Iterate over a copy to allow modification during iteration
            if element.tag in ["vTypeDistribution"]:
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


        # Create the vTypeDistribution element
        vtype_dist_element = ET.Element("vTypeDistribution", id=vtype_dist_id)

        # car_colors = [
        #     "240,240,240",  # Off-White / Light Grey
        #     "40,40,40",     # Dark Grey / Charcoal
        #     "180,180,180",  # Silver / Medium Grey
        #     "25,25,112",    # Midnight Blue / Dark Navy
        #     "128,0,0",      # Maroon / Dark Red
        #     "30,70,30",    # Forest Green / Dark Green
        # ]

        car_colors = [
            "220,220,250",  # Pastel Blue
            "255,240,245",  # Lavender Blush
            "255,255,200",  # Light Yellow / Cream
            "240,230,140",  # Khaki
            "250,170,170",  # Light Coral / Pastel Red
            "152,251,152",  # Pale Green / Mint
        ]

        # Add a vType for each color in the provided list
        for i, color_str in enumerate(car_colors):
            vtype_id = f"color_{i}" 
            
            vtype_attrs = {
                "id": vtype_id,
                "color": color_str,
            }
            
            # Create the vType element with its attributes and add it to the vTypeDistribution
            ET.SubElement(vtype_dist_element, "vType", vtype_attrs)

        # Insert the complete vTypeDistribution element into the root
        root.insert(first_route_index, vtype_dist_element)

        # Write the modified XML tree to the specified output file.
        ET.indent(root, space="    ", level=0)
        tree.write(rou_file_path, encoding="utf-8", xml_declaration=True)
        print(f"Successfully added colorful vTypeDistribution to '{rou_file_path}'")

    except FileNotFoundError:
        print(f"Error: Input file '{rou_file_path}' not found. Please ensure the file exists at the specified path.")
    except ET.ParseError as e:
        print(f"Error parsing XML file '{rou_file_path}': {e}. Please check if the XML is well-formed and valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def color_flows(flows_xml_file_path, vtype_dist_id):
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

