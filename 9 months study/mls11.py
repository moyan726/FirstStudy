import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assumed data for Xi'an marathon routes, landmarks, accommodations, and restaurants
# Format: [longitude, latitude, capacity/gain value]

# Landmarks (must-pass nodes)
landmarks = np.array([
    [108.952, 34.273],  # Landmark 1
    [108.959, 34.274],  # Landmark 2
    [108.951, 34.272],  # Landmark 3
    [108.943, 34.270]  # Landmark 4
])

# Accommodation facilities
accommodation = np.array([
    [108.954, 34.272, 3500],  # Accommodation 1 (location, capacity)
    [108.957, 34.271, 2800],  # Accommodation 2 (location, capacity)
    [108.960, 34.273, 5000]  # Accommodation 3 (location, capacity)
])

# Restaurant facilities (gain value)
restaurants = np.array([
    [108.952, 34.274, 0.2],  # Restaurant 1 (location, gain value)
    [108.956, 34.272, 0.2],  # Restaurant 2 (location, gain value)
    [108.959, 34.275, 0.2]  # Restaurant 3 (location, gain value)
])

# Metro stations
metro_stations = np.array([
    [108.951, 34.275],  # Metro station 1
    [108.960, 34.276]  # Metro station 2
])

# Route data in Xi'an (from point A to point B)
routes = np.array([
    [108.952, 34.273, 108.957, 34.271],  # Route 1: Start and End (longitude, latitude)
    [108.954, 34.272, 108.959, 34.274],  # Route 2: Start and End (longitude, latitude)
    [108.960, 34.273, 108.950, 34.270]  # Route 3: Start and End (longitude, latitude)
])


# Distance function (distance between geographic coordinates in km)
def distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 6371 * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))


# Calculate distance, accommodation capacity, and proximity to metro stations for each route
distances = np.zeros(routes.shape[0])
accommodation_capacity = np.zeros(routes.shape[0])
near_metro = np.zeros(routes.shape[0])

for i in range(routes.shape[0]):
    # Calculate the distance between start and end points
    lat1, lon1, lat2, lon2 = routes[i, 0], routes[i, 1], routes[i, 2], routes[i, 3]
    distances[i] = distance(lat1, lon1, lat2, lon2)

    # Check accommodation capacity within 3 km of the start point
    accommodation_capacity[i] = 0
    for j in range(accommodation.shape[0]):
        if distance(lat1, lon1, accommodation[j, 0], accommodation[j, 1]) <= 3:  # 3 km
            accommodation_capacity[i] += accommodation[j, 2]  # Accumulate capacity

    # Check proximity to metro stations within 1 km
    near_metro[i] = 0
    for j in range(metro_stations.shape[0]):
        if distance(lat1, lon1, metro_stations[j, 0], metro_stations[j, 1]) <= 1:  # 1 km
            near_metro[i] = 1

# Filter valid routes that meet the criteria
valid_routes = np.where((distances >= 42) & (accommodation_capacity >= 3000) & (near_metro == 1))[0]

# Create output table
output_data = {
    'Route': [f'Route {i + 1}' for i in range(routes.shape[0])],
    'Start_Longitude': routes[:, 0],
    'Start_Latitude': routes[:, 1],
    'End_Longitude': routes[:, 2],
    'End_Latitude': routes[:, 3],
    'Distance_km': distances,
    'Accommodation_Capacity': accommodation_capacity,
    'Near_Metro': near_metro
}
output_table = pd.DataFrame(output_data)
output_table['Valid'] = output_table.index.isin(valid_routes)

# Visualization: Scatter plot of landmarks, accommodation, restaurants, and metro stations
plt.figure(figsize=(8, 6))
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', label='Landmarks', marker='o')
plt.scatter(accommodation[:, 0], accommodation[:, 1], c='green', label='Accommodation', marker='s')
plt.scatter(restaurants[:, 0], restaurants[:, 1], c='red', label='Restaurants', marker='^')
plt.scatter(metro_stations[:, 0], metro_stations[:, 1], c='magenta', label='Metro Stations', marker='*')
plt.title('Landmarks, Accommodation, Restaurants, and Metro Stations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()

# Output data to Excel
output_filename = 'Marathon_Route_Analysis.xlsx'
output_table.to_excel(output_filename, index=False)

print(f'The results have been saved to {output_filename}')