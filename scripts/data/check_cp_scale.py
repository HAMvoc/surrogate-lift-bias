"""Check the Cp scale (E0 no. 6): stagnation point at the car nose, and C_D,p integrated from pressure vs total Cd.
Usage: python scripts/data/check_cp_scale.py data/raw"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image, ImageDraw

U, P_INF = 30.0, 0.0
Q = 0.5 * U**2  # p kinematic: Cp = (p - p_inf) / (0.5 U^2)


def frontal_area(tri_pts, px=0.005):
    """Area projected onto the yz plane (m^2) by rasterising; tri_pts: (n,3,3). Grid covers ±1.2 m laterally, 2 m high."""
    w, h = int(2.4 / px), int(2.0 / px)
    img = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(img)
    yz = ((tri_pts[:, :, 1:3] + [1.2, 0.0]) / px)
    for t in yz:
        d.polygon([tuple(p) for p in t], fill=1)
    return np.asarray(img).sum() * px * px


def main(raw):
    raw = Path(raw)
    cd = pd.read_csv(raw / "drag" / "DrivAerNetPlusPlus_Cd_8k_Updated.csv").set_index("ID")["Drag_Value"]
    rows = []
    for f in sorted((raw / "pressure" / "PressureVTK").glob("*.vtk")):
        m = pv.read(f).triangulate()
        pts, faces = m.points, m.faces.reshape(-1, 4)[:, 1:]
        tri = pts[faces]
        # winding is inconsistent in some files: let VTK make it consistent and orient normals outward
        mn = m.compute_normals(cell_normals=True, point_normals=False, consistent_normals=True,
                               auto_orient_normals=True, split_vertices=False)
        n = np.asarray(mn.cell_data["Normals"])
        area = np.asarray(m.compute_cell_sizes(length=False, volume=False)["Area"])
        p = m.point_data["p"]
        pc = p[faces].mean(1)
        cp = (p - P_INF) / Q
        i = int(np.argmax(cp))
        fx = -(pc * area * n[:, 0]).sum()  # drag force / rho
        a_ref = frontal_area(tri)
        cdp = 2 * fx / (U**2 * a_ref)
        rows.append(dict(id=f.stem, cp_max=cp.max(), cp_min=cp.min(), nose_dx=pts[i, 0] - pts[:, 0].min(),
                         nose_z=pts[i, 2], frac_gt1=(cp > 1.0).mean(), a_ref=a_ref,
                         cd_p=cdp, cd_total=cd.get(f.stem, np.nan), cut_open=abs((area[:, None] * n).sum(0)).max()))
    df = pd.DataFrame(rows)
    df["ratio"] = df.cd_p / df.cd_total
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    print("\nratio cd_p/cd_total: mean", round(df.ratio.mean(), 3), "std", round(df.ratio.std(), 3))


if __name__ == "__main__":
    main(sys.argv[1])
