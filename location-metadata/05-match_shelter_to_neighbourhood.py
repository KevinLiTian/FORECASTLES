# Preamble
# Purpose: Match neighbourhoods to shelters using geographical coordinates


# Import packages
import pandas as pd
import numpy as np
import shapely
from shapely import Point
from shapely.geometry import shape
import geojson

if __name__ == "__main__":
    # Load the data
    shelters = pd.read_csv('shelter_locations.csv')

    # Create dictionary of neighbourhood names to multipolygon mappings
    with open('Neighbourhoods - 4326.geojson') as f:
        neighbourhoods = geojson.load(f)
        d = {}
        for nb in neighbourhoods["features"]:
            d[nb["properties"]["AREA_NAME"]] = shape(nb["geometry"])
        print(d)

    # Create Points from latitude and longitude of shelters
    shelter_locs = []
    for row in shelters.to_dict('records'):
        shelter_locs.append(Point(row['LON'], row['LAT']))

    # Match shelters to neighbourhoods
    shelter_neighbourhoods = [""]*len(shelters)
    for nb in d.keys():
        multipolygon = d[nb]
        # check what shelters lie in this neighbourhood
        arr = shapely.contains(multipolygon, shelter_locs)
        s_indices = np.where(arr)[0]
        for ind in s_indices:
            shelter_neighbourhoods[ind] = nb
    shelters['Neighbourhood'] = shelter_neighbourhoods

    # Save to csv
    shelters.to_csv('shelter_neighbourhoods.csv', index=False)

