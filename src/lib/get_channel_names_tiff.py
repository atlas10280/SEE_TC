import tifffile
import xml.etree.ElementTree as ET

def get_channel_names_tiff(tiff_path):
    with tifffile.TiffFile(tiff_path) as tif:
        # Parse the OME-XML metadata
        ome_metadata = tif.ome_metadata
        root = ET.fromstring(ome_metadata)

        # Find all Channel names
        namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}  # may need to adjust version
        channels = root.findall('.//ome:Channel', namespaces)
        channel_names = [ch.attrib.get('Name') for ch in channels]

    return(channel_names)