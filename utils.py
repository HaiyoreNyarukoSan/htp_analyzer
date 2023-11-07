import pickle
from pathlib import Path
from typing import Any, Tuple

from settings import STATIC_DIRS


# Static files
def find_static_file(path: Path):
    for static_dir in STATIC_DIRS:
        if (fullpath := Path(static_dir) / path).exists():
            return fullpath


# Pickle
def fetch_pickle(path: Path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res


def write_pickle(data: Any, path: Path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# BoundingBoxes
def xyxy_in_xyxy(window_xyxy, roof_xyxy, target):
    windows_x1, windows_y1, windows_x2, windows_y2 = window_xyxy
    roof_x1, roof_y1, roof_x2, roof_y2 = roof_xyxy
    r1 = ('x' not in target) or (windows_x1 >= roof_x1 and windows_x2 <= roof_x2)
    r2 = ('y' not in target) or (windows_y1 >= roof_y1 and windows_y2 <= roof_y2)
    return int(r1 and r2)


def xyxy_lt_xyxy(window_xyxy, roof_xyxy, target):
    windows_x1, windows_y1, windows_x2, windows_y2 = window_xyxy
    roof_x1, roof_y1, roof_x2, roof_y2 = roof_xyxy
    r1 = ('x' not in target) or (windows_x2 < roof_x1)
    r2 = ('y' not in target) or (windows_y2 < roof_y1)
    return int(r1 and r2)


def xyxy_gt_xyxy(window_xyxy, roof_xyxy, target):
    windows_x1, windows_y1, windows_x2, windows_y2 = window_xyxy
    roof_x1, roof_y1, roof_x2, roof_y2 = roof_xyxy
    r1 = ('x' not in target) or (windows_x2 > roof_x1)
    r2 = ('y' not in target) or (windows_y2 > roof_y1)
    return int(r1 and r2)


def xyxy_in_xyxys(window_xyxy, roofs_xyxy, options: Tuple[Tuple[str, str]]):
    for roof_xyxy in roofs_xyxy:
        res = 1
        for option in options:
            operation, target = option
            op = operation if operation in ('in', 'lt', 'gt') else 'in'
            func = {'in': xyxy_in_xyxy, 'lt': xyxy_lt_xyxy, 'gt': xyxy_gt_xyxy}[op]
            if not func(window_xyxy, roof_xyxy, target):
                res = 0
                break
        if res: return 1
    return 0


def xyxys_in_xyxys(windows_xyxy, roofs_xyxy, options: Tuple[Tuple[str, str]]):
    return sum(xyxy_in_xyxys(window_xyxy, roofs_xyxy, options) for window_xyxy in windows_xyxy)
