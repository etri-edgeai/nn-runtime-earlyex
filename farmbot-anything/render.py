import os 
import yaml
import pandas as pd
import open3d as o3d
import numpy as np

# Set environment variables
# os.environ['EGL_PLATFORM'] = 'surfaceless'   # Ubuntu 20.04+
# os.environ['OPEN3D_CPU_RENDERING'] = 'true'  # Ubuntu 18.04

# Load config file
cfg_path = "./config/base.yml"
with open(cfg_path,'r') as f:
    cfg = yaml.safe_load(f)

# Define camera positions and orientations

# camera_positions = [
#     [0, 0, 1],  # front
#     [0, 0, -1],  # back
#     [0, 1, 0],  # top
#     [0, -1, 0],  # bottom
#     [1, 0, 0],  # right
#     [-1, 0, 0],  # left
#     [0.5, 0.5, 0.5],  # isometric
#     [-0.5, 0.5, 0.5],  # isometric
#     [0.5, -0.5, 0.5],  # isometric
#     [-0.5, -0.5, 0.5],  # isometric
#     [0.5, 0.5, -0.5],  # isometric
#     [-0.5, 0.5, -0.5],  # isometric
# ]

# Define camera lookat points
lookat_points = [
    [1, 0, 0],  # front, back, top, bottom, left, right
    [1, 0, 0],  # front, back, top, bottom, left, right
    [1, 0, 0],  # front, back, top, bottom, left, right
    [1, 0, 0],  # front, back, top, bottom, left, right
    [1, 0, 0],  # front, back, top, bottom, left, right
    [1, 0, 0],  # front, back, top, bottom, left, right
    [1, 0, 0],  # isometric
    [1, 0, 0],  # isometric
    [1, 0, 0],  # isometric
    [1, 0, 0],  # isometric
    [1, 0, 0],  # isometric
    [1, 0, 0],  # isometric
]
# Define camera up vectors
front_vectors = [
    [0, -1, 0],  # front, back, top, bottom, left, right
    [0, -1, 0],  # front, back, top, bottom, left, right
    [0, -1, 0],  # front, back, top, bottom, left, right
    [0, -1, 0],  # front, back, top, bottom, left, right
    [0, -1, 0],  # front, back, top, bottom, left, right
    [0, -1, 0],  # front, back, top, bottom, left, right
    [0, -1, 0],  # isometric
    [0, -1, 0],  # isometric
    [0, -1, 0],  # isometric
    [0, -1, 0],  # isometric
    [0, -1, 0],  # isometric
    [0, -1, 0],  # isometric
]
# Define camera up vectors
up_vectors = [
    [0, 0, -1],  # front, back, top, bottom, left, right
    [0, 0, -1],  # front, back, top, bottom, left, right
    [0, 0, -1],  # front, back, top, bottom, left, right
    [0, 0, -1],  # front, back, top, bottom, left, right
    [0, 0, -1],  # front, back, top, bottom, left, right
    [0, 0, -1],  # front, back, top, bottom, left, right
    [0, 0, -1],  # isometric
    [0, 0, -1],  # isometric
    [0, 0, -1],  # isometric
    [0, 0, -1],  # isometric
    [0, 0, -1],  # isometric
    [0, 0, -1],  # isometric
]
# up_vectors = [
#     [0, 1, 0],  # front, back, top, bottom, left, right
#     [0, 1, 0],  # front, back, top, bottom, left, right
#     [0, 0, -1],  # front, back, top, bottom, left, right
#     [0, 0, 1],  # front, back, top, bottom, left, right
#     [0, 1, 0],  # front, back, top, bottom, left, right
#     [0, 1, 0],  # front, back, top, bottom, left, right
#     [0.5, 0.5, 0],  # isometric
#     [-0.5, 0.5, 0],  # isometric
#     [0.5, -0.5, 0],  # isometric
#     [-0.5, -0.5, 0],  # isometric
#     [0.5, 0, -0.5],  # isometric
#     [-0.5, 0, -0.5],  # isometric
# ]


if __name__ == "__main__":
    # Set device for PyTorch
    # device = 'cuda:0'
    # device = 'cpu'
    # if not o3d._build_config['ENABLE_HEADLESS_RENDERING']:
    #     print("Headless rendering is not enabled. "
    #           "Please rebuild Open3D with ENABLE_HEADLESS_RENDERING=ON")
    #     exit(1)

    # Load metadata
    metadata_df = pd.read_csv(cfg['shapenet']['metadata'])

    # Choose an item from the metadata
    index = 2
    item = metadata_df.iloc[index]
    img_id = item['fullId']
    category = item['category']
    print(item)
    # Set paths for obj, mtl, binvox, screen, and texture files
    obj_path = os.path.join(cfg['shapenet']['obj'], img_id[4:]) + '.obj'
    mtl_path = os.path.join(cfg['shapenet']['obj'], img_id[4:]) + '.mtl'
    binvox_path = os.path.join(cfg['shapenet']['binvox'], img_id[4:])+'.binvox'
    screen_path = os.path.join(cfg['shapenet']['screen'], img_id[4:])
    screen_path = os.path.join(screen_path, img_id[4:])+ '-0.png'
    # texture_path = os.path.join(cfg['shapenet']['texture'], img_id[4:])

    # Load mesh from obj file
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.textures = [o3d.geometry.Image(mtl_path)]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()
    # ctr.set_lookat([0, 0, 0])
    # ctr.set_front([0, 0, -1])
    # ctr.set_up([0, 1, 0])
    # ctr.set_zoom(1)
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    print(camera_params.extrinsic)
    print("Camera position: ", camera_params.extrinsic[:3, 3])
    print("Camera orientation: ", camera_params.extrinsic[:3, :3])
    print(dir(vis))
    print(dir(ctr))
    # Modify the extrinsic matrix to change the camera position and orientation
    # extrinsic = np.copy(camera_params.extrinsic)
    # extrinsic[:3, :3] = np.eye(3)  # Set the rotation matrix to the identity matrix
    # extrinsic[:3, 3] = [0, 0, 0.5]  # Set the translation vector to [0, 0, 0.5]
    # camera_params.extrinsic = extrinsic

    # Set the camera parameters
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    for i in range(10):
        ctr.set_lookat([1,0,0])
        # ctr.set_front(front_vectors[i])
        # ctr.set_up(up_vectors[i])
        # ctr.set_zoom(0.5)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"render_{i}.png")

    vis.destroy_window()