import numpy as np
import math

def compute_distance_between_points(p1, p2):
    """ 
        Computes distance between two points
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.hypot(dx, dy)


def is_point_in_segment(start_seg, end_seg, point_q):
    """ 
        Determines in point_q is strictly inside the segment defined by start_seg and end_seg
    """
    dist_1 = compute_distance_between_points(start_seg, point_q)
    dist_2 = compute_distance_between_points(end_seg, point_q)
    dist_3 = compute_distance_between_points(start_seg, end_seg)
     
    return dist_1 + dist_2 == dist_3


def compute_lines_intersection(line_1, line_2):
    
    """ 
    The orthogonal projection of a point onto a line is defined as the
    intersection between two perpendicular lines (with the point of
    interest being along one of the lines)
    
    This orthogonal projection can be obtained by finding the inteserction
    point between 2 perpendicular lines
       
    Lines are defined in standard form ax + by + c = 0
    line_1 = <a, b, c>
    line_2 = <a1, b1, c1>
    """
    d  = line_1[0] * line_2[1] - line_1[1] * line_2[0]
    dx = line_1[2] * line_2[1] - line_1[1] * line_2[2]
    dy = line_1[0] * line_2[2] - line_1[2] * line_2[0]
    if d != 0:
        x = dx / d
        y = dy / d
        return np.array([x,y])
    return False


def compute_line_through_points(p1, p2):
   
    """ 
    Computes line defined by 2 points
    Line is returned in standard form ax + by + c = 0
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # If p1 and p2 are different, compute the line defined by these 2 points
    if np.all(np.isclose(p1, p2)):
        return False
    
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0]*p2[1] - p2[0]*p1[1]

    return [a, b, -c]


def compute_distance_point_to_line_by_intersection(start_line, end_line, point_q):
    
    """
    Computes distance from point_q and the line defined by 
    (start_line, end_line).
    
    This corresponds to finding the closest point (x0,y0) on the line to point_q
    (i.e, orthogonal projection of point_q onto the line) and computing
    the distance between (x0,y0) and point_q
    
    (x0, y0) coordinates are computed by considering the intersection between
    two perpendicular lines. The first line is defined by the points start_line
    and end_line. The second line is perpendicular to the first one and goes through
    point_q
    
    """

    # Put point_q in right format
    point_q = np.array(point_q)
    
    # Compute first line parameters
    first_line = compute_line_through_points(start_line, end_line)
           
    if first_line:
        a, b, c = first_line
        
        # Normalize parameters so that a*a + b*b = 1
        ab_norm = np.linalg.norm(np.array([a, b]))
        a_norm = a / ab_norm
        b_norm = b / ab_norm
        
        # Define a norm vector perpendicular to first line
        # This vector is parallel to the line that goes through point_q
        normal = np.array([a_norm, b_norm])

        # Using the vector-form equation of a line, to find a second in line 2
        point_q2 = point_q + 2*(normal)
        
        # Using point_q and point_q2, find the standard parameters of second line
        second_line = compute_line_through_points(point_q, point_q2)
        if second_line:

            # Compute intersection point
            proj_point = compute_lines_intersection(first_line, second_line)
            
            if proj_point is not None:
                # Compute distance between point_q and its projection onto first line
                distance = compute_distance_between_points(proj_point, point_q)
                return proj_point, distance
            
    return False


def compute_distance_point_to_segment(start_seg, end_seg, point_q):
    """
    Computes distance from point_q and line segment defined by start_seg and end_seg
    
    This method first computes the distance (and othorgonal projection of point_q) 
    to the line defined by start_seg and end_seg.
    
    If proj_point_q is stricly inside the segment, it returns the distance
    to the line and the indicator w=1
    
    If proj_point_q is not in the segment, it determines the closest segment
    point to point_q, computes the distance between point_q and the chosen segment point
    
    The indicator will be set to w=1 if start_seg is the closest. Otherwise,
    end_seg is the closest point and w=2
    
    """
    
    # Compute point_q projection and distance from point_q to projection
    proj_q, distance = compute_distance_point_to_line_by_intersection(start_seg, end_seg, point_q)
    
    # Compute distance to start and end of segment
    dist_to_start = compute_distance_between_points(start_seg, point_q)
    dist_to_end = compute_distance_between_points(end_seg, point_q)
    
    if is_point_in_segment(start_seg, end_seg, proj_q):
        w = 0  # orthogonal projection is the closest point
    elif dist_to_start < dist_to_end:
        w = 1 # start is the closest point in segment
        distance = dist_to_start
        proj_q = start_seg
    else:
        w = 2 # end is the closest point in segment
        distance = dist_to_end
        proj_q = end_seg
                
    return w, distance, proj_q


def get_direction_from_points(p1, p2):
    """
    Computes horizontal angle between line defined by p1 and p2 and world x-axis
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.atan2(dy, dx)
    return np.array([np.cos(angle), np.sin(angle)])


def get_direction_from_line(line):
    """
    Computes direction of vector parallel to standard line ax + by + c = 0
    """
    a, b, c = line
    vector = np.array([b, -a])
    vector = vector / np.linalg.norm(vector)
    return vector


def point_in_line(line, point):
    """
    Determines if a point is strictly inside a line
    """
    a, b, c = line
    return np.isclose(a*point[0] + b*point[1] - c, 0)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.linalg.norm(array - value, axis=1)).argmin()
    return idx, array[idx]

def polygonArea(X, Y, n):
 
    # Initialize area
    area = 0.0
 
    # Calculate value of shoelace formula
    j = n - 1
    for i in range(n):
        area = area + (X[j] + X[i]) * (Y[j] - Y[i])
        j = i   # j is previous vertex to i
    return int(abs(area / 2.0))
