import numpy as np 
from numpy.linalg import norm
from typing import List
import math as m
import warnings

# Thanks to https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/ for the code
def _Rx(theta):
    return np.matrix([
        [1, 0           , 0           ],
        [0, m.cos(theta), -m.sin(theta)],
        [0, m.sin(theta), m.cos(theta)]])


def _Ry(theta):
    return np.matrix([
        [m.cos(theta), 0, m.sin(theta)],
        [0           , 1, 0           ],
        [-m.sin(theta), 0, m.cos(theta)]])


def _Rz(theta):
    return np.matrix([
        [m.cos(theta), -m.sin(theta), 0],
        [m.sin(theta), m.cos(theta) , 0],
        [0           , 0            , 1]])


def euler_xyz_to_R(euler):
    """Convert to Rotation matrix.
        Expects input in degrees.
        Only support Maya convension of intrinsic xyz Euler Angles
    """
    return _Rz(np.deg2rad(euler[2])) * _Ry(np.deg2rad(euler[1])) * _Rx(np.deg2rad(euler[0]))



def discretize(params: np.ndarray, bin_size: int, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    params: np.ndarray = (params  - shift) / scale
    params = np.clip(params, 0, 1) * bin_size
    params = params.astype(int).clip(0, bin_size - 1)
    return params

def vector_angle(v1, v2):
    """Find an angle between two 2D vectors"""
    v1, v2 = np.asarray(v1), np.asarray(v2)
    cos = np.dot(v1, v2) / (norm(v1) * norm(v2))
    angle = np.arccos(cos) 
    # Cross to indicate correct relative orienataion of v2 w.r.t. v1
    cross = np.cross(v1, v2)
    
    if abs(cross) > 1e-5:
        angle *= np.sign(cross)
    return angle

def list_to_c(num):
    """Convert 2D list or list of 2D lists into complex number/list of complex numbers"""
    if isinstance(num[0], (list, tuple, set, np.ndarray)):
        return [complex(n[0], n[1]) for n in num]
    else: 
        return complex(num[0], num[1])
# Arcs converters
def is_colinear(start, end, point, eps=1e-5):
    """Check if a point is colinear with the line defined by start and end"""
    return abs(np.cross(end - start, point - start)) < eps

def arc_from_three_points(start, end, point_on_arc):
    """Create a circle arc from 3 points (start, end and any point on an arc)
    
        NOTE: Control point specified in the same coord system as start and end
        NOTE: points should not be on the same line
    """

    nstart, nend, npoint_on_arc = np.asarray(start), np.asarray(end), np.asarray(point_on_arc)

    # https://stackoverflow.com/a/28910804
    # Using complex numbers to calculate the center & radius
    x, y, z = list_to_c([start, point_on_arc, end]) 
    w = z - x
    w /= y - x
    c = (x - y)*(w - abs(w)**2)/2j/w.imag - x
    # NOTE center = [c.real, c.imag]
    rad = abs(c + x)

    # Large/small arc
    mid_dist = norm(npoint_on_arc - ((nstart + nend) / 2))

    # Orientation
    angle = vector_angle(npoint_on_arc - nstart, nend - nstart)  # +/-

    return (start, end, rad, mid_dist > rad, angle > 0) 

def arc_rad_flags_to_three_point(start, end, radius, large_arc, right, local_coordinates=True): 
        """Convert circle to SVG arc parameters"""

        n_start, n_end = np.asarray(start), np.asarray(end)

        # pythagorean theorem for the delta y
        mid_point = (n_start + n_end) / 2
        v = end - n_start
        b = np.linalg.norm(v/2)
        delta_y = np.sqrt(radius**2 - b**2)

        # large arc or not 
        if not large_arc: delta_y *= -1

        y = radius + delta_y

        # sign of y
        y = -y if right else y

        # local coordinates
        if local_coordinates: 
            return start, end, [0.5, y/np.linalg.norm(v)]
    
        # world coordinates
        normal = np.array([-v[1], v[0]])
        normal = normal / np.linalg.norm(normal)

        return start, end, mid_point + y * normal
    
def control_to_abs_coord(start, end, control_scale):
        """
        Derives absolute coordinates of Bezier control point given as an offset
        """
        edge = end - start
        edge_perp = np.array([-edge[1], edge[0]])
        control_start = start + control_scale[0] * edge
        control_point = control_start + control_scale[1] * edge_perp

        return control_point 
    
def control_to_relative_coord(start, end, control_point):
    """
    Derives relative (local) coordinates of Bezier control point given as 
    a absolute (world) coordinates
    """
    start, end, control_point = np.array(start), np.array(end), \
        np.array(control_point)

    control_scale = [None, None]
    edge_vec = end - start
    edge_len = np.linalg.norm(edge_vec)
    control_vec = control_point - start
    
    # X
    # project control_vec on edge_vec by dot product properties
    control_projected_len = edge_vec.dot(control_vec) / (edge_len + 1e-5) 
    control_scale[0] = control_projected_len / (edge_len + 1e-5)
    # Y
    control_projected = edge_vec * control_scale[0]
    vert_comp = control_vec - control_projected  
    control_scale[1] = np.linalg.norm(vert_comp) / (edge_len + 1e-5)
    # Distinguish left&right curvature
    control_scale[1] *= np.sign(np.cross(edge_vec, control_vec))

    return control_scale 

def panel_universal_transtation(vertices: np.ndarray, rotation: List, translation: List):
    """Return a universal 3D translation of the panel (e.g. to be used in judging the panel order).
        Universal translation it defined as world 3D location of mid-point of the top (in 3D) of the panel (2D) bounding box.
        * Assumptions: 
            * In most cases, top-mid-point of a panel corresponds to body landmarks (e.g. neck, middle of an arm, waist) 
            and thus is mostly stable across garment designs.
            * 3D location of a panel is placing this panel around the body in T-pose
        * Function result is independent from the current choice of the local coordinate system of the panel
    """

    # out of 2D bounding box sides' midpoints choose the one that is highest in 3D
    top_right = vertices.max(axis=0)
    low_left = vertices.min(axis=0)
    mid_x = (top_right[0] + low_left[0]) / 2
    mid_y = (top_right[1] + low_left[1]) / 2
    mid_points_2D = [
        [mid_x, top_right[1]], 
        [mid_x, low_left[1]],
        [top_right[0], mid_y],
        [low_left[0], mid_y]
    ]
    rot_matrix = euler_xyz_to_R(rotation)  # calculate once for all points 
    mid_points_3D = np.vstack(tuple(
        [_point_in_3D(coords, rot_matrix, translation) for coords in mid_points_2D]
    ))
    top_mid_point = mid_points_3D[:, 1].argmax()

    return mid_points_3D[top_mid_point], np.array(mid_points_2D[top_mid_point])

def _point_in_3D(local_coord, rotation, translation):
    """Apply 3D transformation to the point given in 2D local coordinated, e.g. on the panel
    * rotation is expected to be given in 'xyz' Euler anges (as in Autodesk Maya) or as 3x3 matrix"""

    # 2D->3D local
    local_coord = np.append(local_coord, 0)

    # Rotate
    rotation = np.array(rotation)
    if rotation.size == 3:  # transform Euler angles to matrix
        rotation = euler_xyz_to_R(rotation)
        # otherwise we already have the matrix
    elif rotation.size != 9:
        raise ValueError('BasicPattern::Error::You need to provide Euler angles or Rotation matrix for _point_in_3D(..)')
    rotated_point = rotation.dot(local_coord)

    # translate
    return rotated_point + translation