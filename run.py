import trimesh
import numpy as np

np.set_printoptions(precision=2, suppress=True)

import os

case = trimesh.load(os.path.join(os.path.dirname(__file__), 'Scotto9 - Case (32U4 Pro Micro).stl'))
ray = trimesh.load(os.path.join(os.path.dirname(__file__), 'rayquaza.stl'))
case_bounds_before = case.bounds.copy()
case_verts_before = case.vertices.copy()

V = ray.vertices
mean = V.mean(axis=0)
Vc = V - mean
cov = np.cov(Vc.T)
eigval, eigvec = np.linalg.eigh(cov)
order = np.argsort(eigval)[::-1]
eigvec = eigvec[:, order]
if np.linalg.det(eigvec) < 0:
    eigvec[:, -1] *= -1
R = eigvec
Perm = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
Rfull = R @ Perm
assert abs(np.linalg.det(Rfull) - 1) < 1e-6

pts, face_idx = trimesh.sample.sample_surface(ray, 200000)
pc1_pts = (pts - mean) @ R[:, 0]
head_point = pts[pc1_pts >= np.percentile(pc1_pts, 95)].mean(axis=0)

S = 0.5
YAW_DEG = 0.0

def full_transform(p, s=S, yaw_deg=YAW_DEG, T=np.zeros(3)):
    q = s * ((p - mean) @ Rfull)
    if yaw_deg != 0:
        th = np.radians(yaw_deg)
        c, sn = np.cos(th), np.sin(th)
        Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        q = q @ Rz.T
    return q + T

V_rs = full_transform(V)
head_rs = full_transform(head_point)
tz = -V_rs[:, 2].min()
tx, ty = -head_rs[0], -head_rs[1]
T = np.array([tx, ty, tz])

V_final = full_transform(V, T=T)
ray_final = trimesh.Trimesh(vertices=V_final, faces=ray.faces, process=False)

print("=== SANITY CHECK: case unchanged? ===")
print("case identical vertices:", np.array_equal(case.vertices, case_verts_before))
print("case bounds:", case.bounds.tolist())

print("\n=== Rayquaza final ===")
print("bounds:\n", ray_final.bounds)
print("is_watertight:", ray_final.is_watertight)
head_final_check = full_transform(head_point, T=T)
print("head final position (want xy=0,0):", head_final_check)
print("total assembly height (case+ray max Z):", max(case.bounds[1,2], ray_final.bounds[1,2]))

ray_final.export('ray_final_v1.stl')

print("\n=== Attempting boolean union ===")
try:
    union = trimesh.boolean.union([case, ray_final], engine='manifold')
    print("union success, watertight:", union.is_watertight, "faces:", len(union.faces))
    union.export('combined_v1_union.stl')
except Exception as e:
    print("union failed:", repr(e))

concat = trimesh.util.concatenate([case, ray_final])
print("concat faces:", len(concat.faces), "watertight:", concat.is_watertight)
concat.export('combined_v1_concat.stl')