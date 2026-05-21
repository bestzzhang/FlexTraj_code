import bpy
import os
import random
import bpy_extras
from mathutils import Vector, Matrix, Euler
import numpy as np
import sys
import math
import argparse

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []  # 没有额外参数

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", default="./")
parser.add_argument("--txt_file", required=True)
args = parser.parse_args(argv)
base_path = args.base_path
txt_file = args.txt_file

with open(txt_file) as f:
    use_models = f.read().splitlines()
print("use_models: ", use_models)

if not os.path.exists(base_path):
    print(f"Error: Specified path does not exist - {base_path}")
    if base_path != default_path:
        print(f"Attempting to use default path: {default_path}")
        if os.path.exists(default_path):
            base_path = default_path
        else:
            print(f"Error: Default path also does not exist")
            sys.exit(1)
    else:
        print(f"Error: Default path does not exist")
        sys.exit(1)

raw_path = os.path.join(base_path, "raw")
if not os.path.exists(raw_path):
    print(f"Error: raw folder does not exist - {raw_path}")
    sys.exit(1)

original_materials = set(bpy.data.materials.keys())
original_images = set(bpy.data.images.keys())

initial_camera_matrix = None
cam = bpy.context.scene.camera
if cam:
    initial_camera_matrix = cam.matrix_world.copy()
else:
    print("Warning: No camera found in the scene")

skybox_path = os.path.join(base_path, "raw/skybox")
if not os.path.exists(skybox_path):
    print(f"Warning: skybox folder does not exist - {skybox_path}")

initial_env_positions = {}

def clean_scene(original_materials, original_images):
    env_collection = bpy.context.scene.collection.children.get('env')
    env_objects = set()
    if env_collection:
        # Restore the positions of env objects
        for obj in env_collection.objects:
            if obj.type == 'EMPTY':
                if obj.name in initial_env_positions:
                    obj.location = initial_env_positions[obj.name].copy()
        env_objects = {obj.name for obj in env_collection.objects}
    
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if obj.name not in env_objects:
            try:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.data.objects.remove(obj, do_unlink=True)
            except ReferenceError:
                continue

    for action in bpy.data.actions:
        try:
            bpy.data.actions.remove(action, do_unlink=True)
        except ReferenceError:
            continue

    current_materials = set(bpy.data.materials.keys())
    imported_materials = current_materials - original_materials
    for material_name in imported_materials:
        try:
            material = bpy.data.materials.get(material_name)
            if material:
                bpy.data.materials.remove(material, do_unlink=True)
        except ReferenceError:
            continue

    current_images = set(bpy.data.images.keys())
    imported_images = current_images - original_images
    for image_name in imported_images:
        try:
            image = bpy.data.images.get(image_name)
            if image:
                bpy.data.images.remove(image, do_unlink=True)
        except ReferenceError:
            continue
            
    import gc
    gc.collect()

def import_fbx(file_path, **options):
    selected_before = set(obj.name for obj in bpy.context.selected_objects)
    active_before = bpy.context.active_object
    
    bpy.ops.import_scene.fbx(filepath=file_path, **options)
    
    new_selected = set(obj.name for obj in bpy.context.selected_objects) - selected_before
    armature = None
    
    for obj_name in new_selected:
        obj = bpy.data.objects.get(obj_name)
        if obj and obj.type == 'ARMATURE':
            armature = obj
            for collection in obj.users_collection:
                collection.objects.unlink(obj)
            avatar_collection = bpy.context.scene.collection.children.get('avatar')
            if avatar_collection:
                avatar_collection.objects.link(obj)
            break
            
    bpy.ops.object.select_all(action='DESELECT')
    for obj_name in selected_before:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.select_set(True)
    if active_before:
        bpy.context.view_layer.objects.active = active_before
        
    return armature

def get_scene_inverse_depth_range(scene, cam):
    """Calculate the inverse depth values of all vertices in the scene and return the 2% and 98% percentiles."""
    inverse_depths = []
    
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
            
        world_matrix = obj.matrix_world
        mesh = obj.data
        
        for vertex in mesh.vertices:
            world_coord = world_matrix @ vertex.co
            distance = (world_coord - cam.matrix_world.translation).length
            inverse_depth = 1.0 / distance if distance > 0 else 0
            inverse_depths.append(inverse_depth)
    
    if not inverse_depths:
        return 0, 1  # Default values
        
    inverse_depths = np.array(inverse_depths)
    min_inverse = np.percentile(inverse_depths, 2)
    max_inverse = np.percentile(inverse_depths, 98)
    
    return min_inverse, max_inverse

def set_vertex_colors_for_object(obj, cam):
    """Set vertex colors and material for a single object."""
    if obj.type != 'MESH':
        return
    
    mesh = obj.data
    if obj.data.shape_keys or obj.modifiers:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)
        mesh_eval = object_eval.data
    else:
        mesh_eval = mesh
    
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="Col")
    
    color_layer = mesh.vertex_colors.active
    
    world_matrix = obj.matrix_world
    
    cam_location = cam.matrix_world.translation
    cam_direction = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    
    counter = 0
    MAX_COLOR_INDEX = 256*256

    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            
            world_coord = world_matrix @ mesh_eval.vertices[vertex_index].co
            
            to_vertex = (world_coord - cam_location).normalized()
            
            is_backface = to_vertex.dot(cam_direction) < 0
            
            if not is_backface:
                if counter < MAX_COLOR_INDEX - 1:
                    # Normal case: calculate color from index
                    r = 1.0
                    g_int, b_int = divmod(counter, 256)
                    g = g_int / 255.0
                    b = b_int / 255.0
                    color = (r, g, b, 1.0)
                else:
                    color = (1.0, 1.0, 1.0, 1.0)
                counter += 1
            
            # If coordinates are on a backface, set to black
            if is_backface:
                color = (0, 0, 0, 1.0)
            
            color_layer.data[loop_index].color = color
    
    if counter > MAX_COLOR_INDEX:
        print(f"ID Overflow Warning:, counter, Max: {MAX_COLOR_INDEX}, Got: {counter}")

    mat_name = f"ScreenSpaceColor_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    nodes.clear()
    
    vertex_color = nodes.new(type='ShaderNodeVertexColor')
    vertex_color.layer_name = "Col"
    
    emission = nodes.new(type='ShaderNodeEmission')
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    links = mat.node_tree.links
    links.new(vertex_color.outputs[0], emission.inputs[0])
    links.new(emission.outputs[0], output.inputs[0])
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def filter_processed_files(animation_file, model_name):
    if model_name.split("_")[1][0] != "t" and animation_file.split("_")[1][0] != "9":
            return True
    return False


class ModelProcessor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_file_path = os.path.join(raw_path, model_name, f"{model_name}.fbx")
        self.animation_folder = os.path.join(raw_path, model_name, "motion")

        self.output_path = os.path.join(base_path, "trackingavatars")
        self.videos_path = os.path.join(self.output_path, "videos")
        self.tracking_path = os.path.join(self.output_path, "tracking")

        # self.depth_path = os.path.join(self.output_path, "depth")
        # self.videos_txt = os.path.join(self.output_path, "videos.txt")
        # self.trackings_txt = os.path.join(self.output_path, "trackings.txt")
        
        self.animation_files = []
        self.model_armature = None
        self.animation_armature = None

    def run(self, num_cam_trajs=1):
        if not os.path.exists(self.animation_folder) or not os.path.exists(self.model_file_path):
            print(f"Missing animation folder or model file.")
            return
        
        self.init_paths_and_files()
        self.record_initial_env_positions()
        self.animation_files = sorted(f for f in os.listdir(self.animation_folder) if f.endswith('.fbx'))

        for i, animation_file in enumerate(self.animation_files):
            if filter_processed_files(animation_file, self.model_name):
                num_cam_trajs = 0
                continue
            else:
                num_cam_trajs = 1

            print(f"Processing {animation_file} as file #{i}")
            save_name = "_".join([self.model_name, animation_file.split('.')[0]])

            if os.path.exists(f"{self.tracking_path}/{save_name}_traj_{num_cam_trajs-1}.mp4"):
                continue
            try:
                # set up scene, e.g. camera...
                cam_start_points, cam_end_points = self.prepare_scene_for_rendering(num_cam_trajs)

                # set up character
                self.import_model_and_animation(animation_file)
                self.bind_animation_to_model()
                
                # set up render resolution
                self.setup_render_settings(animation_file)

                # no need to use ground plane
                self.hide_plane()

                
                for j, (cam_start_point, cam_end_point) in enumerate(zip(cam_start_points, cam_end_points)):
                    # set up random background
                    self.setup_random_background()

                    self.set_camera_traj(cam_start_point, cam_end_point)

                    # rendering
                    self.render_main_video(f"{save_name}_traj_{j}")
                    self.render_tracking_video(f"{save_name}_traj_{j}")
                    self.reset_camera_traj()
                
            except Exception as e:
                import traceback
                print(f"[ERROR] Failed to process {animation_file}: {e}")
                traceback.print_exc()
            finally:
                # clean up
                clean_scene(original_materials, original_images)
            

    def init_paths_and_files(self):
        for path in [self.videos_path, self.tracking_path]:
            os.makedirs(path, exist_ok=True)
        # for txt in [self.videos_txt, self.trackings_txt]:
        #     if not os.path.exists(txt):
        #         open(txt, "a").close()

    def record_initial_env_positions(self):
        env_collection = bpy.context.scene.collection.children.get('env')
        if env_collection:
            for obj in env_collection.objects:
                if obj.type == 'EMPTY' and obj.name not in initial_env_positions:
                    initial_env_positions[obj.name] = obj.location.copy()

    def import_model_and_animation(self, animation_file):
        animation_path = os.path.join(self.animation_folder, animation_file)
        self.animation_armature = import_fbx(
            animation_path,
            use_manual_orientation=True,
            axis_forward='-Z',
            axis_up='Y',
            use_anim=True,
            bake_space_transform=True
        )
        self.model_armature = import_fbx(
            self.model_file_path,
            use_manual_orientation=True,
            axis_forward='-Z',
            axis_up='Y'
        )
        if not self.animation_armature or not self.model_armature:
            raise RuntimeError("FBX import failed")

    def bind_animation_to_model(self):
        if bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_remove()

        for obj in bpy.data.objects:
            if obj.rigid_body:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_remove()
            if obj.rigid_body_constraint:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.constraint_remove()
            if obj.type == 'MESH':
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object == self.model_armature:
                        mod.object = self.animation_armature

        self.model_armature.hide_render = True
        self.model_armature.hide_viewport = True

    def render_main_video(self, save_name):
        output_file = f"{save_name}.mp4"

        bpy.context.scene.render.filepath = os.path.join(self.videos_path, output_file)
        bpy.ops.render.render(animation=True)

    def render_tracking_video(self, save_name):
        scene = bpy.context.scene
        cam = scene.camera
        if not cam:
            print("No camera found.")
            return
        env_collection = scene.collection.children.get('env')
        env_objs = {obj.name for obj in env_collection.objects} if env_collection else set()
        character_meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.name not in env_objs]

        if not character_meshes:
            print("No character meshes found.")
            return

        # plane_mesh = next((o for o in scene.objects if o.name == 'plane' and o.type == 'MESH'), None)
        # inverse_depth_range = get_scene_inverse_depth_range(scene, cam)
        original_materials = {obj: obj.data.materials[0] for obj in character_meshes if obj.data.materials}
        # plane_original = plane_mesh.data.materials[0] if plane_mesh and plane_mesh.data.materials else None

        for mesh in character_meshes:
            set_vertex_colors_for_object(mesh, cam)
            
        # if plane_mesh:
        #     mat_name = f"ScreenSpaceColor_{plane_mesh.name}"
        #     if not bpy.data.materials.get(mat_name):
        #         set_vertex_colors_for_object(plane_mesh, scene, cam, inverse_depth_range)
        #     else:
        #         plane_mesh.data.materials[0] = bpy.data.materials.get(mat_name)

        # Set black background and render tracking video
        world = bpy.data.worlds['World']
        world.use_nodes = True
        nodes = world.node_tree.nodes
        nodes.clear()
        background = nodes.new('ShaderNodeBackground')
        output = nodes.new('ShaderNodeOutputWorld')
        world.node_tree.links.new(background.outputs[0], output.inputs[0])
        background.inputs[0].default_value = (0, 0, 0, 1)
                
        tracking_file = f"{save_name}.mp4"
        bpy.context.scene.render.filepath = os.path.join(self.tracking_path, tracking_file)
        bpy.ops.render.render(animation=True)

        for obj, mat in original_materials.items():
            obj.data.materials[0] = mat
        # if plane_mesh and plane_original:
        #     plane_mesh.data.materials[0] = plane_original

        for obj in character_meshes:
            mat = bpy.data.materials.get(f"ScreenSpaceColor_{obj.name}")
            if mat:
                bpy.data.materials.remove(mat, do_unlink=True)

    def render_depth_video(self, save_name: str, depth_max_dist: float = 20.0):
        scene = bpy.context.scene
        
        original_use_nodes = scene.use_nodes
        original_pass_z = scene.view_layers["ViewLayer"].use_pass_z
        original_film_transparent = scene.render.film_transparent
        original_color_mode = scene.render.image_settings.color_mode
        
        try:
            scene.render.film_transparent = True
            scene.view_layers["ViewLayer"].use_pass_z = True
            scene.use_nodes = True
            tree = scene.node_tree
            links = tree.links
            
            for node in tree.nodes:
                tree.nodes.remove(node)
                
            # ---- 创建节点图 ----
            render_layer_node = tree.nodes.new('CompositorNodeRLayers')
            
            map_range_node = tree.nodes.new(type="CompositorNodeMapRange")
            map_range_node.inputs['From Min'].default_value = 0.0
            map_range_node.inputs['From Max'].default_value = depth_max_dist

            invert_node = tree.nodes.new('CompositorNodeInvert')

            composite_node = tree.nodes.new('CompositorNodeComposite')
            
            links.new(render_layer_node.outputs['Depth'], map_range_node.inputs['Value'])
            links.new(map_range_node.outputs['Value'], invert_node.inputs['Color'])
            links.new(invert_node.outputs['Color'], composite_node.inputs['Image'])

            output_file = f"{save_name}.mp4"
            scene.render.filepath = os.path.join(self.depth_path, output_file)

            bpy.ops.render.render(animation=True)

        finally:
            # print("Cleaning up compositor and restoring original render settings.")
            if scene.use_nodes:
                for node in scene.node_tree.nodes:
                    scene.node_tree.nodes.remove(node)
            
            scene.use_nodes = original_use_nodes
            scene.view_layers["ViewLayer"].use_pass_z = original_pass_z
            scene.render.film_transparent = original_film_transparent
            scene.render.image_settings.color_mode = original_color_mode

    @staticmethod
    def prepare_scene_for_rendering(num_cam_trajs=3):
        clean_scene(original_materials, original_images)
        cam = bpy.context.scene.camera
        if cam and initial_camera_matrix is not None:
            cam.matrix_world = initial_camera_matrix.copy()

            # if env := bpy.context.scene.collection.children.get('env'):
            #     for obj in env.objects:
            #         if obj.type == 'EMPTY':
            #             obj.location.z += offset * 0.1
            #             obj.location.x += x_offset
        
        avatar_collection = bpy.context.scene.collection.children.get('avatar')
        if not avatar_collection:
            avatar_collection = bpy.data.collections.new('avatar')
            bpy.context.scene.collection.children.link(avatar_collection)

        if not cam:
            return []

        loc0 = cam.location.copy()
        rot0 = cam.rotation_euler.copy()
        
        # Prepare endpoints list
        start_points = []
        end_points = []
        
        # add trajs
        for i in range(num_cam_trajs):
            z_offset = random.uniform(0, 0.7)
            y_offset = random.uniform(0, 4)
            x_offset = random.uniform(-1, 1)
            y_rotate = random.uniform(-35.0, 35.0) if i > 0 else 0

            s_loc = Vector((loc0.x + x_offset, loc0.y - y_offset, loc0.z + z_offset))
            start_points.append((s_loc, rot0.copy()))

            e_rot = rot0.copy()
            e_rot.y += math.radians(y_rotate)
            end_points.append((s_loc, e_rot))

        return start_points, end_points

    @staticmethod
    def setup_random_background():
        world = bpy.data.worlds['World']
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        
        nodes.clear()
        
        background = nodes.new('ShaderNodeBackground')
        output = nodes.new('ShaderNodeOutputWorld')
        links.new(background.outputs[0], output.inputs[0])
        
        skybox_files = [f for f in os.listdir(skybox_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if skybox_files:
            tex_coord = nodes.new('ShaderNodeTexCoord')
            mapping = nodes.new('ShaderNodeMapping')
            texture = nodes.new('ShaderNodeTexEnvironment')
            
            random_rotation = random.uniform(0, 2 * 3.14159)
            mapping.inputs['Rotation'].default_value[2] = random_rotation
            
            links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], texture.inputs['Vector'])
            
            random_image = random.choice(skybox_files)
            img_path = os.path.join(skybox_path, random_image)
            texture.image = bpy.data.images.load(img_path)
            
            links.new(texture.outputs[0], background.inputs[0])
            
            return os.path.splitext(random_image)[0]
    
    @staticmethod
    def setup_render_settings(animation_file, is_animation=True):
        scene = bpy.context.scene
        
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences
        prefs.addons['cycles'].preferences.compute_device_type = 'OPTIX'  # Or 'OPTIX' for RTX cards
        
        for device in prefs.addons['cycles'].preferences.devices:
            device.use = True
        
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        
        scene.eevee.taa_render_samples = 16
        
        scene.render.resolution_x = 720
        scene.render.resolution_y = 480
        scene.render.fps = 24
        
        if is_animation:
            scene.render.image_settings.file_format = 'FFMPEG'
            scene.render.ffmpeg.format = 'MPEG4'
            scene.render.ffmpeg.codec = 'H264'

            actions = [a for a in bpy.data.actions if a.users > 0]
            if actions:
                action = actions[-1]  # latest in list with users
                frame_start, raw_frame_end = int(action.frame_range[0]), int(action.frame_range[1])
                if raw_frame_end < 49:
                    with open("blender_debug_log.txt", "a") as f:
                        f.write(f"{animation_file} - Expect: 49; Actual: {raw_frame_end - frame_start}\n")
                    print(f"{animation_file} - Expect: 49; Actual: {raw_frame_end - frame_start}")
                frame_end = min(max(frame_start + 49, raw_frame_end), frame_start + 120)
                scene.frame_start = frame_start
                scene.frame_end = frame_end
                print(f"[INFO] Set frame range: {frame_start} → {frame_end}")
            else:
                scene.frame_start = 1
                scene.frame_end = 49
        else:
            scene.render.image_settings.file_format = 'PNG'

    @staticmethod
    def hide_plane():
        plane_mesh = None
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH' and obj.name == 'plane':
                plane_mesh = obj
                break
        if plane_mesh:
            plane_mesh.hide_render = True
            plane_mesh.hide_viewport = True

    @staticmethod
    def set_camera_traj(cam_start_point, cam_end_point):
        """
        Set camera keyframes at scene.frame_start and scene.frame_end.
        cam_start_point: Tuple[Vector, Euler] – starting location and rotation
        cam_end_point: Tuple[Vector, Euler] – ending location and rotation
        """
        
        scene = bpy.context.scene
        cam = scene.camera
        if not cam:
            print("No camera found.")
            return

        for con in cam.constraints:
            con.mute = True

        frame_start = scene.frame_start
        frame_end = scene.frame_end

        # Set and keyframe starting position
        start_loc, start_rot = cam_start_point
        cam.location = start_loc
        cam.rotation_euler = start_rot
        cam.keyframe_insert(data_path="location", frame=frame_start)
        cam.keyframe_insert(data_path="rotation_euler", frame=frame_start)

        # Set and keyframe ending position
        end_loc, end_rot = cam_end_point
        cam.location = end_loc
        cam.rotation_euler = end_rot
        cam.keyframe_insert(data_path="location", frame=frame_end)
        cam.keyframe_insert(data_path="rotation_euler", frame=frame_end)

    @staticmethod
    def reset_camera_traj():
        """
        Completely remove all keyframes and animation data from the camera.
        """
        cam = bpy.context.scene.camera
        if cam and cam.animation_data:
            cam.animation_data_clear()
        for con in cam.constraints:
            con.mute = False

def main():
    model_folders = []
    for item in sorted(os.listdir(raw_path)):
        if os.path.isdir(os.path.join(raw_path, item)) and item != "skybox":
            if item in use_models:
                model_folders.append(item)
    
    if not model_folders:
        print("Error: No model folders found")
        return
    
    print(f"Found the following model folders: {', '.join(model_folders)}")
    
    for model_name in model_folders:
        try:
            processor = ModelProcessor(model_name)
            processor.run()
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue

main() 



