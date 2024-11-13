import logging
import os
import shutil
import numpy as np
import itertools
import math

from rasterio._io import Resampling
from tqdm import tqdm
import xarray as xr
import rioxarray

from src.shadowspy.dem_processing import prepare_dem_mesh
from render_dem import render_at_date
from src.shadowspy.coord_tools import unproject_stereographic
from src.mesh_operations.mesh_utils import import_mesh

RAYTRACING_BACKEND = 'cgal' # 'embree' # 'cgal' #
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
parser.add_argument('--pole', type=str, default="south", choices=["south","north"])

parser.add_argument('--illum_type', type=str, default="lowsunele", choices=["lowsunele","highsunele","fibonacci"])

parser.add_argument('--column', type=int, default=0)

parser.add_argument('--resolution', type=int, default=20) # in meters
parser.add_argument('--patch_size', type=int, default=96) # in pixels

args = parser.parse_args()


# compute direct flux from the Sun
Fsun = 1361  # W/m2
Rb = 1737.4 # km
base_resolution = args.resolution

# constants
n_sun_angles = 2*30
phi = (math.sqrt(5)+1)/2 - 1
ga = phi*2*math.pi

# download kernels
# download_kernels()
path_to_furnsh = "/home/mrepasky3/shadowspy/examples/aux/simple.furnsh"

# Elevation/DEM GTiff input
if args.pole == "south":
    lonlat0 = (0, -90)
    scale_factor = 1
elif args.pole == "north":
    lonlat0 = (0, 90)
    scale_factor = 1e3 # scale to km

tif_path = args.tif_path
patches_dir = "{}_{}_{}px_{}MPP".format(args.pole, args.illum_type, args.patch_size, args.resolution)

class Opt:
    def __init__(self, base_resolution=base_resolution/scale_factor, fartopo_path=None, bbox_roi=False, siteid=0, max_extension=1.,
        mesh_ext='.vtk', lonlat0_stereo=lonlat0, extres={}):
        self.base_resolution = base_resolution
        self.fartopo_path = fartopo_path
        self.bbox_roi = bbox_roi
        self.siteid = siteid
        self.max_extension = max_extension
        self.mesh_ext = mesh_ext
        self.lonlat0_stereo = lonlat0_stereo
        self.extres = extres

if not os.path.exists(patches_dir):
    os.mkdir(patches_dir)

# sun angles
if args.illum_type == "highsunele":
    azi_ele_list = []
    for az_ in np.arange(0,360,10):
        azi_ele_list.append([az_,45])
    for az_ in np.arange(0,360,20):
        azi_ele_list.append([az_,60])
    for az_ in np.arange(0,360,30):
        azi_ele_list.append([az_,75])
    for az_ in np.arange(0,360,60):
        azi_ele_list.append([az_,85])
    azi_ele_list = np.array(azi_ele_list)
    np.save("{}/azi_ele.npy".format(patches_dir), azi_ele_list)
elif args.illum_type == "lowsunele":
    az_ = np.arange(0,360,5)
    el_ = np.array([5,10,20,30])
    azi_ele_list = np.array(list(itertools.product(az_, el_)))
    np.save("{}/azi_ele.npy".format(patches_dir), azi_ele_list)

# patch meters is width of each patch
patch_meters = args.resolution * args.patch_size / scale_factor

# padding is necessary to avoid edge effects of rendering
padding = args.resolution * 1.5  / scale_factor

# patch radius is half the width plus padding
patch_radius = (patch_meters/2) + padding

hd_dem = xr.open_dataarray(tif_path)

x0 = hd_dem[0].x[0].data
x1 = hd_dem[0].x[-1].data

y0 = hd_dem[0].y[0].data
y1 = hd_dem[0].y[-1].data

# define the coordinates at the edges
x_min = min(x0, x1) + patch_radius
x_max = max(x0, x1) - patch_radius

y_min = min(y0, y1) + patch_radius
y_max = max(y0, y1) - patch_radius

patches_per_row = (x_max-x_min)//patch_meters
patches_per_column = (y_max-y_min)//patch_meters


# iterate through a column of patches
patch_counter = args.column*patches_per_column

x_it = x_min + patch_meters*args.column

for j in range(int(patches_per_column)):
    patch_top_dir = "{}/patch_{:06d}".format(patches_dir, int(patch_counter))
    
    if not os.path.exists(patch_top_dir):
        os.mkdir(patch_top_dir)
    elif os.path.exists(f'{patch_top_dir}/imgs_patch.tif'):
        patch_counter += 1
        continue

    opt = Opt(siteid=int(patch_counter))

    hd_dem_patch = hd_dem.rio.clip_box(
        minx=x_it - patch_radius,
        miny=(y_min+patch_meters*j) - patch_radius,
        maxx=x_it + patch_radius,
        maxy=(y_min+patch_meters*j) + patch_radius,
    )

    opt.dem_path = f'{patch_top_dir}/hd_patch.tif'
    hd_dem_patch.rio.to_raster(opt.dem_path)
    dem_crs = str(hd_dem_patch.rio.crs)

    # prepare mesh of the input dem
    opt.tmpdir = f"{patch_top_dir}/tmp/"
    os.makedirs(opt.tmpdir, exist_ok=True)

    inner_mesh_path, outer_mesh_path, dem_path = prepare_dem_mesh(opt.dem_path, opt.tmpdir, opt.siteid, opt)

    lon, lat = unproject_stereographic(x_it, y_min+patch_meters*j, lonlat0[0], lonlat0[1], R=Rb*1e3)
    np.save("{}/lon_lat.npy".format(patch_top_dir), np.array([lon,lat]))

    # import hr meshes and build shape_models
    V_st, F_st, N_st, P_st = import_mesh(f"{inner_mesh_path}_st{opt.mesh_ext}", get_normals=True, get_centroids=True)
    V, F, N, P = import_mesh(f"{inner_mesh_path}{opt.mesh_ext}", get_normals=True, get_centroids=True)

    shape_model = MyTrimeshShapeModel(V, F, N)

    if args.illum_type == "fibonacci":
        # generate sun angles
        i_= np.arange(args.n_sun_angles/2)
        az_= (ga*i_)/(2*math.pi)
        az_ = 360*(az_ - np.floor(az_))
        el_ = -np.arcsin(-1 + 2*i_/args.n_sun_angles)*180/math.pi

        az_ += 3.* np.random.randn(az_.shape[0])
        el_ += 3.* np.random.randn(el_.shape[0])

        azi_ele_list = np.concatenate([az_.reshape(-1,1),el_.reshape(-1,1)],axis=-1)
        np.save("{}/azi_ele.npy".format(patch_top_dir), azi_ele_list)

    # get list of images from mapprojected folder
    img_list = []
    for idx, azi_ele in enumerate(azi_ele_list):
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
    img_list.flux.rio.to_raster("{}/imgs_patch.tif".format(patch_top_dir)) 
    # img_list.flux.rio.to_raster("{}/imgs_patch.tif".format(patch_top_dir), compress='zstd')

    shutil.rmtree(opt.tmpdir, ignore_errors=True)
    # os.remove(opt.dem_path)

    patch_counter += 1