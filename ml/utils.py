# ml/utils.py
import numpy as np

def parse_custom_points(txt):
    """
    Parse lines of x1,x2,y  OR x1,x2,x3,y
    """
    pts = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(',')]
        if len(parts) not in (3,4):
            raise ValueError("Each custom line must be x1,x2,y OR x1,x2,x3,y")
        if len(parts) == 3:
            x1 = float(parts[0]); x2 = float(parts[1]); y = int(float(parts[2]))
            pts.append((np.array([x1,x2]), int(y)))
        else:
            x1 = float(parts[0]); x2 = float(parts[1]); x3 = float(parts[2]); y = int(float(parts[3]))
            pts.append((np.array([x1,x2,x3]), int(y)))
    return pts

def is_3d_dataset(dataset):
    if len(dataset) == 0:
        return False
    first = dataset[0][0]
    return len(first) == 3
