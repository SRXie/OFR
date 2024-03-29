# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
from itertools import product, combinations
import numpy as np

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.
This file expects to be run from Blender like this:
blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.")
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='clevr_obj_test/image_generation/data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='clevr_obj_test/image_generation/data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='clevr_obj_test/image_generation/data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='clevr_obj_test/image_generation/data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=6, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")
parser.add_argument('--size_correlated', action='store_true',
    help="If the sizes of objects are correlated.")
parser.add_argument('--shape_correlated', action='store_true',
    help="If the shapes of objects are correlated.")
parser.add_argument('--color_correlated', action='store_true',
    help="If the colors of objects are correlated.")
parser.add_argument('--material_correlated', action='store_true',
    help="If the materials of objects are correlated.")
parser.add_argument('--appr_correlated', action='store_true',
    help="If the appearence (color+material) of objects are correlated.")
parser.add_argument('--intr_correlated', action='store_true',
    help="If the intrinsic attributes (color+material+size+shape) of objects are correlated.")
parser.add_argument('--pos_margin', default=0.0, type=float,
    help="The margin for positional clustering.")
parser.add_argument('--mutex', action='store_true',
    help="If the color of object clusters are mutually exclusive.")
parser.add_argument('--en_margin', default=0, type=int,
    help="The margin for hard cap on energy.")
parser.add_argument('--en_sigma', default=0.0, type=float,
    help="The sigma of the energy term")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='/checkpoint/siruixie/clevr_corr/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='/checkpoint/siruixie/clevr_corr/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='/checkpoint/siruixie/clevr_corr/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='/checkpoint/siruixie/clevr_corr/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=240, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

def insert_corr_to_dir(dir_path, scene_file=False):
  dir_list = dir_path.split("/")
  index = -1 if scene_file else -2
  if args.pos_margin>0.0:
    if args.mutex:
      dir_list.insert(index, "pos_mutex_"+str(args.pos_margin))
    else:
      dir_list.insert(index, "pos_corr_"+str(args.pos_margin))
  elif args.intr_correlated:
    if args.en_sigma > 0.0:
      dir_list.insert(index, "intr_corr_"+str(args.en_sigma))
    else:
      dir_list.insert(index, "intr_corr_"+str(args.en_margin))
  elif args.appr_correlated:
    if args.en_sigma > 0.0:
      dir_list.insert(index, "appr_corr_"+str(args.en_sigma))
    else:
      dir_list.insert(index, "appr_corr_"+str(args.en_margin))
  else:
    corr_name = ""
    if args.size_correlated:
      corr_name += ("size_"+str(args.en_margin))
    if args.material_correlated:
      corr_name += ("material_"+str(args.en_margin))
    if args.shape_correlated:
      corr_name += ("shape_"+str(args.en_margin))
    if args.color_correlated:
      if args.en_sigma > 0.0:
        corr_name += ("color_"+str(args.en_sigma))
      else:
        corr_name += ("color_"+str(args.en_margin))
    if len(corr_name) > 0:
      dir_list.insert(index, corr_name)
    else:
      dir_list.insert(index, "iid")
  dir_path = "/".join(dir_list)
  return dir_path

def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)

  args.output_image_dir=insert_corr_to_dir(args.output_image_dir)
  args.output_scene_dir=insert_corr_to_dir(args.output_scene_dir)
  args.output_blen_dir=insert_corr_to_dir(args.output_blend_dir)
  args.output_scene_file=insert_corr_to_dir(args.output_scene_file, True)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)

  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  all_scene_paths = []
  for i in range(args.num_images):
    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    num_objects = random.randint(args.min_objects, args.max_objects)
    render_scene(args,
      num_objects=num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
    )

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)

def stirling2 (n, m):
  """ computes the Stirling numbers of the second kind.
    S2(N,M) represents the number of distinct partitions of N elements
    into M nonempty sets.  For a fixed N, the sum of the Stirling
    numbers S2(N,M) is represented by B(N), called "Bell's number",
    and represents the number of distinct partitions of N elements.
  Parameters:
    Input, integer N, the number of rows of the table.
    Input, integer M, the number of columns of the table.
    Output, integer S2(N,M), the Stirling numbers of the second kind."""

  s2 = np.zeros ( ( n, m ) )

  if ( n <= 0 ):
    return s2

  if ( m <= 0 ):
    return s2

  s2[0,0] = 1
  for j in range ( 1, m ):
    s2[0,j] = 0

  for i in range ( 1, n ):

    s2[i,0] = 1

    for j in range ( 1, m ):
      s2[i,j] = ( j + 1 ) * s2[i-1,j] + s2[i-1,j-1]

  return s2[n-1, m-1]

def generate_distr_dict(num_obj, num_val, sigma=0.0, margin=0):
  from itertools import product, combinations
  energy_tensor = np.zeros([num_val]*num_obj)
  distr_dict = {}
  for vals in product(range(num_val), repeat=num_obj):
    if sigma == 0.0:
      valid = True
      for pair in combinations(vals, 2):
          valid = valid and abs(pair[0]-pair[1])<=margin
          if valid:
            energy_tensor[vals] = 1.0
          else:
            energy_tensor[vals] = 0.0
            break
    elif num_obj> 1:
      if num_val == 2:
        energy_tensor[vals] += np.exp(-(sum(vals)-num_obj)**2/(2*math.factorial(num_obj)/math.factorial(num_obj-num_val)*(num_obj/6.0*sigma)**2))
      else:
        # we define the energy over the count of unique values in the axis and then distribute this energy equally to all its possible partitions
        energy_tensor[vals] += np.exp(-(len(set(vals))-1)**2/(2*stirling2(num_obj, len(set(vals)))*(num_obj/6.0*sigma)**2))
  if num_obj> 1:
    energy_tensor = energy_tensor/energy_tensor.sum()
  else:
    energy_tensor = (energy_tensor+1.0)/num_val
  for vals in product(range(num_val), repeat=num_obj):
    distr_dict["-".join(str(v) for v in vals)] = energy_tensor[vals]
  return distr_dict

def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  bg_mapping = {}
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    for name, rgb in properties['bg_color'].items():
      rgba = [float(c) / 255.0 for c in rgb]+[1.0]
      bg_mapping[name] = rgba

  color_name, rgba = random.choice(list(bg_mapping.items()))
  bpy.data.objects["Ground"].color = rgba
  gt_mat = bpy.data.materials.new("bgcolor")
  gt_mat.use_nodes = True
  diffuse_bsdf = gt_mat.node_tree.nodes["Diffuse BSDF"]
  diffuse_bsdf.inputs[0].default_value = rgba
  # gt_mat.use_object_color = True
  bpy.data.objects["Ground"].active_material = gt_mat

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_random_objects(scene_struct, num_objects, args, camera):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['color'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['material'].items()]
    object_mapping = [(v, k) for k, v in properties['shape'].items()]
    size_mapping = list(properties['size'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  obj_cluster = {}
  size_names = []
  obj_names = []
  obj_name_outs = []
  clusters = []
  cluster_means = []
  blender_objects = []

  for attr in ("shape", "size"):
    if eval("args."+attr+"_correlated"):
      exec("global "+attr+"_distr_"+str(num_objects))
      # exec("tmp=np.random.choice(list("+attr+"_distr_"+str(num_objects)+".keys()), p=list("+attr+"_distr_"+str(num_objects)+".values()))", globals())
      exec("correlated_"+attr+"_list=np.random.choice(list("+attr+"_distr_"+str(num_objects)+".keys()), p=list("+attr+"_distr_"+str(num_objects)+".values())).split('-')", globals())

  for i in range(num_objects):
    # Choose a random size
    if not args.size_correlated:
      size_name, r = random.choice(size_mapping)
    else:
      size_name, r = size_mapping[int(correlated_size_list[i])]

    # Choose random shape
    if not args.shape_correlated:
      obj_name, obj_name_out = random.choice(object_mapping)
    else:
      obj_name, obj_name_out = object_mapping[int(correlated_shape_list[i])]

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera)
      while True:
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        if i > 0 and args.pos_margin>0.0 and num_objects<5 and size_names[0] == 'large':
            x = positions[0][0]+x*args.pos_margin
            y = positions[0][1]+y*args.pos_margin
        if i > 0 and args.pos_margin>0.0 and num_objects==5 and size_names[0] == 'large':
            x = positions[0][0]+x*0.8
            y = positions[0][1]+y*0.8
        elif i > 0 and args.pos_margin>0.0 and size_names[0] == 'small':
            x = positions[0][0]+x*args.pos_margin/2.0
            y = positions[0][1]+y*args.pos_margin/2.0
        if x > -3 and x < 3 and y > -3 and y < 3:
          break
      # Check to make sure the new object is further than min_dist from all
      # other objects, and further than margin along the four cardinal directions
      corners_good = True
      dists_good = True
      margins_good = True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if x + y + 3.5 < 0 or x + y - 3.5 >0:
          corners_good = False
          break
        if dist - r - rr < args.min_dist:
          dists_good = False
          break
        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < args.margin:
            print(margin, args.margin, direction_name)
            print('BROKEN MARGIN!')
            margins_good = False
            break
        if not margins_good:
          break

      if corners_good and dists_good and margins_good:
        break

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # now we register the new object to object clusters
    if len(cluster_means) == 0:
      cluster_means.append((1, x, y))
      clusters.append([i])
      obj_cluster[i] = len(clusters)-1
    else:
      cluster_found = False
      dist_min = 100.0
      cluster_min_idx = -1
      for idx, (n, xm, ym) in enumerate(cluster_means):
        dx, dy = x - xm, y - ym
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < dist_min:
          dist_min = dist
          cluster_min_idx = idx
      # if dist_min - r < args.pos_margin: # TODO: adjust with pixel coordinate
      #   cluster_means[cluster_min_idx] = (n+1, (n*xm+x)/(n+1), (n*ym+y)/(n+1))
      #   clusters[cluster_min_idx].append(i)
      #   obj_cluster[i] = cluster_min_idx
      #   cluster_found = True
      if not cluster_found:
        cluster_means.append((1, x, y))
        clusters.append([i])
        obj_cluster[i] = len(clusters)-1

    positions.append((x, y, r))
    size_names.append(size_name)
    obj_names.append(obj_name)
    obj_name_outs.append(obj_name_out)
    # if args.pos_margin == 0.0:
    clusters = [[item for cluster in clusters for item in cluster]]
    # print(clusters, "-----------------------")

  # if args.pos_margin> 0.0 and args.mutex:
  #   correlated_color_lists = np.random.permutation(8)[:len(clusters)]

  for k, cluster in enumerate(clusters):
    for attr in ("material", "color"):
      if eval("args."+attr+"_correlated"):
        exec("global "+attr+"_distr_"+str(len(cluster)))
        exec("correlated_"+attr+"_list=np.random.choice(list("+attr+"_distr_"+str(len(cluster))+".keys()), p=list("+attr+"_distr_"+str(len(cluster))+".values())).split('-')", globals())
    # if args.pos_margin> 0.0 and args.mutex:
    #   global correlated_color_list
    #   correlated_color_list = [correlated_color_lists[k]]*len(cluster)
    for j, obj_idx in enumerate(cluster):
      # Choose random color
      if shape_color_combos is None:
        if not args.color_correlated:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))
        else:
          color_name, rgba = list(color_name_to_rgba.items())[int(correlated_color_list[j])]
      else:
        raise NotImplementedError
        # obj_name_out, color_choices = random.choice(shape_color_combos)
        # color_name = random.choice(color_choices)
        # obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        # rgba = color_name_to_rgba[color_name]

      # Attach a random material
      if not args.material_correlated:
        mat_name, mat_name_out = random.choice(material_mapping)
      else:
        mat_name, mat_name_out = material_mapping[int(correlated_material_list[j])]

      # Choose random orientation for the object.
      theta = 360.0 * random.random()

      # Actually add the object to the scene
      utils.add_object(args.shape_dir, obj_names[obj_idx], positions[obj_idx][2], (positions[obj_idx][0], positions[obj_idx][1]), theta=theta)
      obj = bpy.context.object
      blender_objects.append(obj)

      utils.add_material(mat_name, Color=rgba)

      # Record data about the object in the scene data structure
      pixel_coords = utils.get_camera_coords(camera, obj.location)
      objects.append({
        'shape': obj_name_outs[obj_idx],
        'size': size_names[obj_idx],
        'material': mat_name_out,
        '3d_coords': tuple(obj.location),
        'rotation': theta,
        'pixel_coords': pixel_coords,
        'color': color_name,
      })

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_random_objects(scene_struct, num_objects, args, camera)

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.
  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = render_args.use_antialiasing

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
  utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # Move the lights and ground back to layer 0
  utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
  utils.set_layer(bpy.data.objects['Ground'], 0)

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  render_args.use_antialiasing = old_use_antialiasing

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    if args.pos_margin > 0.0 or args.appr_correlated:
      args.en_margin = 0
      args.color_correlated = True
      args.material_correlated = True
    if args.intr_correlated:
      args.en_margin = 0
      args.color_correlated = True
      args.material_correlated = True
      args.shape_correlated = True
      args.size_correlated = True
    if args.shape_correlated:
      for i in range(args.max_objects):
        exec("shape_distr_"+str(i+1)+"=generate_distr_dict(i+1, 3, 0.0, margin=args.en_margin)")
    if args.size_correlated:
      for i in range(args.max_objects):
        exec("size_distr_"+str(i+1)+"=generate_distr_dict(i+1, 2, args.en_sigma, margin=args.en_margin)")
    if args.material_correlated:
      for i in range(args.max_objects):
        exec("material_distr_"+str(i+1)+"=generate_distr_dict(i+1, 2, args.en_sigma, margin=args.en_margin)")
    if args.color_correlated:
      for i in range(args.max_objects):
        exec("color_distr_"+str(i+1)+"=generate_distr_dict(i+1, 8, args.en_sigma, margin=args.en_margin)")
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')
