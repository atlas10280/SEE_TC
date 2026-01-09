# -*- coding: utf-8 -*-

# given the metadata string with exposure times, parse them into a df by channel
def parse_exposure_times(input_string, c_names):
    import re
    import pandas as pd
    # Initialize dictionary
    parsed_dict = {}

    # Parse the header (e.g., Flash4.0, SN:002010)
    header_match = re.match(r"^(.*),\s*SN:(\S+)", input_string)
    if header_match:
        parsed_dict["Device"] = header_match.group(1).strip()
        parsed_dict["SerialNumber"] = header_match.group(2).strip()

    # Parse the samples
    sample_pattern = re.compile(
        r"Sample (\d+):\r?\n\s+Exposure: ([\d.]+) (\w+)\r?\n\s+Binning: (\S+)\r?\n\s+Scan Mode: (\w+)"
    )

    samples = {}
    for match in sample_pattern.finditer(input_string):
        sample_id = int(match.group(1))
        samples[sample_id] = {
            "Exposure": {"Value": float(match.group(2)), "Unit": match.group(3)},
            "Binning": match.group(4),
            "ScanMode": match.group(5),
        }

    parsed_dict["Samples"] = samples

    # Create a new dictionary with updated keys
    parsed_dict["Samples"] = {new_key: samples[old_key] for new_key, old_key in zip(c_names, samples.keys())}
    samples_dict = parsed_dict["Samples"]

    # Convert to DataFrame
    samples_df = pd.DataFrame.from_dict(samples_dict, orient='index').reset_index()
    samples_df.rename(columns={"index": "channel_ID"}, inplace=True)
    return(parsed_dict,samples_df)