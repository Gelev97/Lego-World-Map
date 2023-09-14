import requests
import cv2
import numpy as np
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import argparse

LEGO_WIDTH = 128  # Adjust as per your LEGO map's width
LEGO_HEIGHT = 80  # Adjust as per your LEGO map's height
IMAGE_PATH = './ori.jpg' # Lego Original Picture
OUTPUT_PATH = './result.jpg' # Lego Marked City Picture

def get_coordinates_osm(place_name):
    """
    Get the latitude and longitude of a place using Nominatim from OpenStreetMap.

    Args:
    - place_name (str): The name of the place.

    Returns:
    - tuple: (latitude, longitude) or None if not found.
    """

    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json"
    }

    headers = {
        "User-Agent": "YourAppName"  # Replace "YourAppName" with a suitable user-agent for your application
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        latitude = float(data['lat'])
        longitude = float(data['lon'])
        return (latitude, longitude)
    return None

def real_to_lego_coords(latitude, longitude, legoWidth, legoHeight):
    """
    Convert real world coordinates (latitude and longitude) to LEGO map coordinates.

    Args:
    - latitude (float): Latitude of the real world coordinate.
    - longitude (float): Longitude of the real world coordinate.
    - legoWidth (int): Width of the LEGO map in studs.
    - legoHeight (int): Height of the LEGO map in studs.

    Returns:
    - tuple: (legoX, legoY) coordinates on the LEGO map.
    """

    # Convert latitude and longitude to percentages (0-1 scale)
    lat_percent = (latitude + 90) / 180
    lon_percent = (longitude + 180) / 360

    # Convert the percentages to LEGO coordinates
    legoX = int(lon_percent * legoWidth)
    legoY = legoHeight - int(lat_percent * legoHeight)  # Subtract from height to get top-down y-coordinates

    return [legoX, legoY]

def detect_circles(image_path, output_path, lego_coords):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    for lego_coord in lego_coords:
        center = (round(lego_coord[0]*2307/128-8), round(lego_coord[1]*1444/80-10))  # center coordinate
        cv2.circle(img, center, 8, (0, 0, 255), 2)  # circle outline
        cv2.circle(img, center, 2, (0, 0, 255), 3)  # circle center

    # Save the result to output_path
    cv2.imwrite(output_path, img)

def calibration():
    with open('./calibration.yaml', 'r') as file:
        calib_locations = yaml.safe_load(file)
    predicted_locations = np.zeros((len(calib_locations), 2))
    correct_locations = np.zeros((len(calib_locations), 2))
    for i, calib_location in enumerate(calib_locations):
        predicted_locations[i] = calib_location['predicted_location']
        correct_locations[i] = calib_location['correct_location']
 
    # Calculate differences between predicted and correct locations (dx and dy)
    dx = correct_locations[:, 0] - predicted_locations[:, 0]
    dy = correct_locations[:, 1] - predicted_locations[:, 1]

    # Create a feature matrix (X) with predicted x and y coordinates
    X = predicted_locations

    # Split the data into training and test sets
    X_train, X_test, dx_train, dx_test, dy_train, dy_test = train_test_split(
        X, dx, dy, test_size=0.2, random_state=42)

    # Train linear regression models for dx and dy
    model_dx = LinearRegression()
    model_dy = LinearRegression()

    model_dx.fit(X_train, dx_train)
    model_dy.fit(X_train, dy_train)

    # Make predictions for dx and dy
    dx_pred = model_dx.predict(np.array([[50, 50]]))
    dy_pred = model_dy.predict(np.array([[50, 50]]))

    return model_dx, model_dy

def main():
    parser = argparse.ArgumentParser(description="With Calibration")
    parser.add_argument("-c", "--calibrate", action='store_true', help="Get output with calibration", required=False)
    args = parser.parse_args()

    if args.calibrate:
        model_dx, model_dy = calibration()
    
    with open('./cities.yaml', 'r') as file:
        places = yaml.safe_load(file)
    
    lego_coords = []
    for place in places:
        coordinates = get_coordinates_osm(place)
        if coordinates:
            print(f"Coordinates for {place}: {coordinates}")
        else:
            print(f"Couldn't find coordinates for {place}")

        lego_coord = real_to_lego_coords(coordinates[0], coordinates[1], 
                                        LEGO_WIDTH, LEGO_HEIGHT)
        if args.calibrate:
            lego_coord[0] = round(lego_coord[0] + model_dx.predict(np.array([lego_coord]))[0])
            lego_coord[1] = round(lego_coord[1] + model_dy.predict(np.array([lego_coord]))[0])
        
        print(f"LEGO Coordinates: {lego_coord}")
        lego_coords.append(lego_coord)
    detect_circles(IMAGE_PATH, OUTPUT_PATH, lego_coords)

if __name__ == "__main__":
    main()