## TODO complete this file with implementation of A* starting from Dijkstra's implementation in Dijkstra.py.
import math

def manhattan_distance(point1, point2):
    """Compute manhattan distance between two points.
    
    Formula:
        h(n) = |x_1 - x_2| + |y_1 - y_2|

    Args:
        point1: A tuple representing the coordinates of the first point (x1, y1).
        point2: A tuple representing the coordinates of the second point (x2, y2).
    Returns:
        The euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean_distance(point1, point2):
    """Compute euclidean distance between two points.

    Formula:
        d = sqrt((x_1 - x_2)^2 + (y_1 - y_2)^2)

    Args:
        point1: A tuple representing the coordinates of the first point (x1, y1).
        point2: A tuple representing the coordinates of the second point (x2, y2).
    Returns:
        The euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2))


def haversine_distance(point1, point2):
    """Compute haversine distance between two points.

    Formula:
        a = sin^2((lat2 - lat1) / 2) + cos(lat1) * cos(lat2) * sin^2((lon2 - lon1) / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        d = R * c

    Args:
        point1: A tuple representing the coordinates of the first point (lat1, lon1).
        point2: A tuple representing the coordinates of the second point (lat2, lon2).
    Returns:
        The haversine distance between the two points in kilometers.
    """
    lat1, lon1 = point1
    lat2, lon2 = point2
    R = 6371

    #phi1 = math.radians(lat1)
    phi1 = lat1
    #phi2 = math.radians(lat2)
    phi2 = lat2
    delta_phi = math.radians(lat1 - lat2)
    delta_lambda = math.radians(lon1 - lon2)

    a = (math.sin(delta_phi / 2) ** 2) + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(a-1))

    return R * c