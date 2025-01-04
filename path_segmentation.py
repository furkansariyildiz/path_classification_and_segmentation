import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.config import threading


tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)


model = load_model('/home/furkan/imu-master-hw/models/model.h5')


def rotateData(data, angle):
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


def generateCurvyRoadForTest(start_x, samples, amplitude, period=10, noise_scale=0.55):
    """
    Generates a curvy road for path segmentation (test side).

    Parameters:
    start_x (float): Start point of the curvy road.
    samples (int): The number of sample points to create the curvy road.
    amplitude (float): The amplitude of the curvy road (sine wave signal).
    period (float, optional): The period of the curvy road (sine wave signal). Default is 10.
    noise_scale (float, optional): The scale of noise to be added using np.random.normal(). Default is 0.55.

    Returns:
    numpy.ndarray: The generated curvy road with sine wave type.
    """
    x = np.linspace(start_x, start_x + 10, int(samples))
    y = amplitude * np.sin(2 * np.pi * (x - start_x) / period) + np.random.normal(scale=noise_scale, size=int(samples))
    return np.column_stack((x, y))



def generateStraightRoadForTest(start_x, samples, noise_scale=0.1):
    """
    Generates a straight road for path segmentation (test side).

    Parameters:
    start_x (float): Start point of the straight road.
    samples (int): The number of sample points to create the straight road.
    noise_scale (float, optional): The scale of noise to be added using np.random.normal(). Default is 0.1.

    Returns:
    numpy.ndarray: The generated straight road.
    """
    x = np.linspace(start_x, start_x + 10, int(samples))
    y = np.zeros(int(samples)) + np.random.normal(scale=noise_scale, size=int(samples))  # Y değerlerini sıfırlarla dolduruyoruz, çünkü yol tamamen düz
    return np.column_stack((x, y))



def generateTriangleWaveForTest(start_x, samples, amplitude, period=10, noise_scale=0.1):
    """
    Generates a triangle wave for path segmentation (test side).

    Parameters:
    start_x (float): Start point of the triangle wave road.
    samples (int): The number of sample points to create the triangle wave.
    amplitude (float): The amplitude of the triangle wave.
    period (float, optional): The period of the triangle wave. Default is 10.
    noise_scale (float, optional): The scale of noise to be added using np.random.normal(). Default is 0.1.

    Returns:
    numpy.ndarray: The generated triangle wave path with noise.
    """
    x = np.linspace(start_x, start_x + 10, int(samples))
    y = amplitude * ((((x - start_x) / period) - 0.5) % 1 - 0.5) + np.random.normal(scale=noise_scale, size=int(samples))
    y = np.where(y < 0, 0, y)
    return np.column_stack((x, y))



def createSequentialTestRoad(samples_per_segment=1000):
    """
    Creates a sequential road for testing path segmentation.

    Parameters:
    samples_per_segment (int): The number of samples for each road segment.

    Returns:
    numpy.ndarray: A vertically stacked array containing all road segments.
    """
    segment1 = generateStraightRoadForTest(0, samples_per_segment, noise_scale=0)
    segment2 = generateCurvyRoadForTest(segment1[-1, 0], samples_per_segment, 1, 10, noise_scale=0)
    segment3 = generateTriangleWaveForTest(segment2[-1, 0], samples_per_segment, 5, 10, noise_scale=0)
    segment4 = generateTriangleWaveForTest(segment3[-1, 0], samples_per_segment, 5, 10, noise_scale=0)

    full_road = np.vstack([segment1, segment2, segment3, segment4])
    return full_road



def detectChanges(data, window_size=100, threshold=0.1):
    """
    Detects changes in the path for segmentation based on variance within a sliding window.

    The function calculates the variance for each sliding window of the specified size and
    checks if the variance exceeds the given threshold. If the variance is higher than the
    threshold, it indicates a change in the road segment.

    Parameters:
    data (numpy.ndarray): The road data to be segmented.
    window_size (int): The size of the sliding window for detecting changes. Default is 100.
    threshold (float): The variance threshold for detecting changes. Default is 0.1.

    Returns:
    list of tuple: A list of tuples where each tuple contains the start and end indices of
                   the window where a change was detected.
    """
    changes = []
    for i in range(0, len(data) - window_size, window_size):
        window = data[i:i + window_size]
        var = np.var(window[:, 1])
        if var > threshold:
            changes.append((i, i + window_size))
    return changes




# Creating and rotating road data.
test_road = createSequentialTestRoad(1000)
test_road_rotated = rotateData(test_road, 0)

# Detecting changes
change_points = detectChanges(test_road_rotated, window_size=1000, threshold=0.75)

# Classification segments
predicted_segments = []
for start, end in change_points:
    segment = test_road_rotated[start:end]
    if len(segment) < 1000:
        padding = np.zeros((1000 - len(segment), 2))
        segment = np.vstack([segment, padding])
    segment = segment.reshape(1, 1000, 2)
    prediction = model.predict(segment)
    predicted_label = np.argmax(prediction)
    predicted_segments.append(predicted_label)

# Predicted labels
labels = ['Straight', 'Curvy', 'Triangle']
predicted_labels = [labels[label] for label in predicted_segments]

# Acctual results
segment_colors = {'Straight': 'blue', 'Curvy': 'green', 'Triangle': 'red'}
colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']

# Plotting
plt.figure(figsize=(12, 6))

used_labels = set()

for i, (start, end) in enumerate(change_points):
    label = predicted_labels[i]  
    color = segment_colors[label] 
    
    if label not in used_labels:
        plt.plot(test_road_rotated[start:end, 0], test_road_rotated[start:end, 1], color=color, label=label)
        used_labels.add(label)
    else:
        plt.plot(test_road_rotated[start:end, 0], test_road_rotated[start:end, 1], color=color)


plt.title('Segmented Road Classification')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


print("Predicted labels:", predicted_labels)
print("Actual labels:", ["Straight", "Curvy", "Triangle"])