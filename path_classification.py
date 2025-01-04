import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os


model_path = os.path.join(os.path.dirname(__file__), 'path_classification_model.h5')
model = load_model(model_path) 



def generate_curvy_road(samples, amplitude, period=10, noise_scale=0.55):
    """
    Generates a curvy road for path classification.

    Parameters:
    samples (int): The number of sample points to create the curvy road.
    amplitude (float): The amplitude of the curvy road (sine wave signal).
    period (float, optional): The period of the curvy road (sine wave signal). Default is 10.
    noise_scale (float, optional): The scale of noise to be added using np.random.normal(). Default is 0.55.

    Returns:
    numpy.ndarray: The generated curvy road with sine wave type.
    """
    x = np.linspace(5, 10, samples)
    y = amplitude * np.sin(2 * np.pi * x / period) + np.random.normal(scale=noise_scale, size=samples)
    return np.column_stack((x, y))



def generate_straight_road(samples, noise_scale=0.1):
    """
    Generates a straight road for path classification.

    Parameters:
    samples (int): The number of sample points to create the straight road.
    noise_scale (float, optional): The scale of noise to be added using np.random.normal(). Default is 0.1.

    Returns:
    numpy.ndarray: The generated straight road.
    """
    x = np.linspace(0, 10, samples)
    y = np.zeros(samples) + np.random.normal(scale=noise_scale, size=samples)
    return np.column_stack((x, y))



def generate_triangle_road(samples, amplitude, period=10, noise_scale=0.1):
    """
    Generates a triangle wave for path classification.

    Parameters:
    samples (int): The number of sample points to create the triangle wave.
    amplitude (float): The amplitude of the triangle wave.
    period (float, optional): The period of the triangle wave. Default is 10.
    noise_scale (float, optional): The scale of noise to be added using np.random.normal(). Default is 0.1.

    Returns:
    numpy.ndarray: The generated triangle wave path with noise.
    """
    x = np.linspace(0, 10, samples)
    y = amplitude * (((x / period) - 0.5) % 1 - 0.5) +  np.random.normal(scale=noise_scale, size=samples)
    y = np.where(y < 0, 0, y)
    return np.column_stack((x, y))



def rotate_data(data, angle):
    """
    Rotates the given road data by the specified angle.

    Parameters:
    data (numpy.ndarray): The road data to be rotated.
    angle (float): The angle in degrees by which to rotate the data.

    Returns:
    numpy.ndarray: The rotated road data.
    """
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return np.dot(data, rotation_matrix)



def create_data_sets(samples=1000):
    """
    Creates datasets for model training.

    Parameters:
    samples (int): The number of samples for each road type.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - x_data (numpy.ndarray): The generated road data.
        - y_data (numpy.ndarray): The corresponding labels for each road type.
          Labels are one-hot encoded:
            - Curvy road: [0, 1, 0]
            - Straight road: [1, 0, 0]
            - Triangle wave: [0, 0, 1]
    """
    x_data = []
    y_data = []

    # Curvy Road
    for count in range(100):
        for angle in range(360):
            rotated_data = rotate_data(generate_curvy_road(samples, (count + 1) * 0.1, (count + 1) * 0.5, (count) * 0.01), angle)
            x_data.append(rotated_data)
            y_data.append([0, 1, 0])  # [0, 1, 0] Curvy

    # Straight Road
    for count in range(100):
        for angle in range(360):
            rotated_data = rotate_data(generate_straight_road(samples, (count) * 0.001), angle)
            x_data.append(rotated_data)
            y_data.append([1, 0, 0])  # [1, 0, 0] Straight

    # Triangle Wave
    for count in range(100):
        for angle in range(360):
            rotated_data = rotate_data(generate_triangle_road(samples, (count + 1) * 0.1, (count + 1) * 0.5, (count) * 0.01), angle)
            x_data.append(rotated_data)
            y_data.append([0, 0, 1])  # [0, 0, 1] Triangle Wave

    return np.array(x_data), np.array(y_data)

# Create data set
x_data, y_data = create_data_sets()


# Creating test values
x_test_samples = np.array([generate_curvy_road(1000, 0.2, 1.0, noise_scale=0),
                           generate_straight_road(1000, noise_scale=0),
                           generate_triangle_road(1000, 5, 5, noise_scale=0)])

# Rotate roads
x_test_samples_rotated = [rotate_data(x_test_samples[i], 0) for i in range(len(x_test_samples))]

# Prediction step
predictions = model.predict(np.array(x_test_samples_rotated))

# Converting prediction to labels
predicted_labels = np.argmax(predictions, axis=1)

# Plotting predictions with labels
labels = ['Straight', 'Curvy', 'Triangle']
for i in range(len(predicted_labels)):
    predicted_label = labels[predicted_labels[i]]
    plt.figure(figsize=(8, 5))
    plt.plot(x_test_samples_rotated[i][:, 0], x_test_samples_rotated[i][:, 1], label='Road')
    plt.title(f"Prediction: {predicted_label}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()