import logging
import os
import shutil
import numpy as np
import itertools


from rasterio._io import Resampling
from tqdm import tqdm
import xarray as xr
import rioxarray

from src.shadowspy.dem_processing import prepare_dem_mesh
from render_dem import render_at_date
from src.shadowspy.coord_tools import project_stereographic
from src.mesh_operations.mesh_utils import import_mesh

RAYTRACING_BACKEND = 'cgal' # 'embree'
RAYTRACING_BACKEND = RAYTRACING_BACKEND.lower()
if RAYTRACING_BACKEND == 'cgal':
    from src.shadowspy.shape import CgalTrimeshShapeModel as MyTrimeshShapeModel
elif RAYTRACING_BACKEND == 'embree':
    try:
        import embree
    except:
        logging.error("* You need to add embree_vars to the PATH to use embree")
        exit()
    from src.shadowspy.shape import EmbreeTrimeshShapeModel as MyTrimeshShapeModel
else:
    raise ValueError('RAYTRACING_BACKEND should be one of: "cgal", "embree"')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tif_path', type=str)

parser.add_argument('--resolution', type=int, default=20) # in meters
parser.add_argument('--patch_size', type=int, default=3840) # in pixels

args = parser.parse_args()


# compute direct flux from the Sun
Fsun = 1361  # W/m2
Rb = 1737.4 # km
base_resolution = args.resolution

# download kernels
# download_kernels()
path_to_furnsh = "/home/mrepasky3/shadowspy/examples/aux/simple.furnsh"

# Elevation/DEM GTiff input
tif_path = args.tif_path
patch_top_dir = "large_region_{}px_{}MPP".format(args.patch_size, args.resolution)
lonlat0 = (0, -90)

class Opt:
    def __init__(self, base_resolution=base_resolution, fartopo_path=None, bbox_roi=False, siteid=0, max_extension=1.,
        mesh_ext='.vtk', lonlat0_stereo=lonlat0, extres={}):
        self.base_resolution = base_resolution
        self.fartopo_path = fartopo_path
        self.bbox_roi = bbox_roi
        self.siteid = siteid
        self.max_extension = max_extension
        self.mesh_ext = mesh_ext
        self.lonlat0_stereo = lonlat0_stereo
        self.extres = extres

opt = Opt(siteid=0)

if not os.path.exists(patch_top_dir):
	os.mkdir(patch_top_dir)

# get sun angles
aziele_partitions = []
aziele_partition_names = ["5","10","20","30","45","60-85"]

# 5 degree elevation
az_ = np.arange(0,360,5)
el_ = np.array([5])
aziele_partitions.append(list(itertools.product(az_, el_)))

# 10 degree elevation
az_ = np.arange(0,360,5)
el_ = np.array([10])
aziele_partitions.append(list(itertools.product(az_, el_)))

# 20 degree elevation
az_ = np.arange(0,360,5)
el_ = np.array([20])
aziele_partitions.append(list(itertools.product(az_, el_)))

# 30 degree elevation
az_ = np.arange(0,360,5)
el_ = np.array([30])
aziele_partitions.append(list(itertools.product(az_, el_)))

# 45 degree elevation
azi_ele_list = []
for az_ in np.arange(0,360,10):
    azi_ele_list.append([az_,45])
aziele_partitions.append(azi_ele_list)

# 60 - 85 degree elevation
azi_ele_list = []
for az_ in np.arange(0,360,20):
    azi_ele_list.append([az_,60])
for az_ in np.arange(0,360,30):
    azi_ele_list.append([az_,75])
for az_ in np.arange(0,360,60):
    azi_ele_list.append([az_,85])
aziele_partitions.append(azi_ele_list)

# prepare DEM patch
hd_dem = xr.open_dataarray(tif_path)
xc, yc = project_stereographic(0, -85, lonlat0[0], lonlat0[1], R=Rb*1e3)

patch_meters = args.resolution * args.patch_size 	# width of each patch
padding = args.resolution * 1.5 					# avoids edge effects of rendering
patch_radius = (patch_meters/2) + padding

hd_dem_patch = hd_dem.rio.clip_box(
	minx=xc - patch_radius,
	miny=yc - patch_radius,
	maxx=xc + patch_radius,
	maxy=yc + patch_radius,
)

opt.dem_path = f'{patch_top_dir}/hd_patch.tif'
if not os.path.exists(opt.dem_path):
	hd_dem_patch.rio.to_raster(opt.dem_path)


# prepare mesh of the input dem
dem_crs = str(hd_dem_patch.rio.crs)

opt.tmpdir = f"{patch_top_dir}/tmp/"
os.makedirs(opt.tmpdir, exist_ok=True)

inner_mesh_path, outer_mesh_path, dem_path = prepare_dem_mesh(opt.dem_path, opt.tmpdir, opt.siteid, opt)

# import hr meshes and build shape_models
V_st, F_st, N_st, P_st = import_mesh(f"{inner_mesh_path}_st{opt.mesh_ext}", get_normals=True, get_centroids=True)
V, F, N, P = import_mesh(f"{inner_mesh_path}{opt.mesh_ext}", get_normals=True, get_centroids=True)

shape_model = MyTrimeshShapeModel(V, F, N)

# render images
for k in range(len(aziele_partitions)):
	azi_ele_list = np.array(aziele_partitions[k])
	np.save("{}/azi_ele_{}.npy".format(patch_top_dir, aziele_partition_names[k]), azi_ele_list)

	img_list = []
	for idx, azi_ele in tqdm(enumerate(azi_ele_list), total=len(azi_ele_list), desc='render'):
		azi_ele = tuple(azi_ele)
		
		dsi, _ = render_at_date(meshes=None,
	            path_to_furnsh=path_to_furnsh, crs=dem_crs, inc_flux=Fsun, show=False, azi_ele_deg=azi_ele,
	            V_st=V_st, F_st=F_st, N_st=N_st, P_st=P_st, V=V, F=F, N=N, P=P, shape_model=shape_model)
		
		dsi.rio.write_crs(dem_crs, inplace=True)
		target_resolution = int(round(hd_dem_patch.rio.resolution()[0],0))
		dsi.rio.reproject_match(hd_dem_patch, resampling=Resampling.bilinear, inplace=True)
		dsi = dsi.rio.reproject(dsi.rio.crs, resolution=target_resolution, resampling=Resampling.bilinear)

		dsi = dsi.assign_coords(time=idx)
		dsi = dsi.expand_dims(dim="time")
		img_list.append(dsi.astype(np.float32))
	img_list = xr.combine_by_coords(img_list)
	img_list.flux.rio.to_raster("{}/imgs_patch_{}.tif".format(patch_top_dir, aziele_partition_names[k]))

	shutil.rmtree(opt.tmpdir, ignore_errors=True)