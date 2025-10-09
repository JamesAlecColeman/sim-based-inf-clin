import numpy as np


LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

NEIGHBOURS_FACE = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])  # 6
NEIGHBOURS_EDGE = np.array([(-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),  # 12
                            (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
                            (0, -1, -1), (0, 1, -1), (0, -1, 1), (0, 1, 1)])
NEIGHBOURS_CORNER = np.array([(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (1, 1, 1),
                              (1, 1, -1), (-1, 1, -1), (1, -1, 1), (1, -1, -1)])  # 8
NEIGHBOURS_26 = np.concatenate((NEIGHBOURS_FACE, NEIGHBOURS_EDGE, NEIGHBOURS_CORNER))  # 26