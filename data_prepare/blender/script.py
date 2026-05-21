import bpy
import bpy_extras
from mathutils import Vector
import numpy as np

def set_vertex_colors_for_object(obj, cam, r=1.0):
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
    
    # color_layer = mesh.vertex_colors.active
    color_layer = mesh.vertex_colors.get("Col")
    
    world_matrix = obj.matrix_world
    
    cam_location = cam.matrix_world.translation
    cam_direction = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    
    counter = 0
    MAX_COLOR_INDEX = 256*100

    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            
            world_coord = world_matrix @ mesh_eval.vertices[vertex_index].co
            
            to_vertex = (world_coord - cam_location).normalized()
            
            is_backface = to_vertex.dot(cam_direction) < 0
            
            if not is_backface:
                if counter < MAX_COLOR_INDEX - 1:
                    # Normal case: calculate color from index
                    g_int, b_int = divmod(counter, 256)
                    g = g_int / 255.0
                    b = b_int / 255.0
                    color = (r, g, b, 1.0)
                else:
                    counter = 0
                    color = (r+0.004, 0.0, 0.0, 1.0)
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


def process_all_objects():
    """Process all mesh objects in the scene"""
    scene = bpy.context.scene
    cam = scene.camera
    
    if not cam:
        print("No camera in scene!")
        return
    
    count = 0
    for obj in scene.objects:
        if obj.type == 'MESH':
            count += 1
    colors = [i / count for i in range(count + 1)]
    print("count: ", count)
    
    count = 0
    for obj in scene.objects:
        if obj.type == 'MESH':
            print(f"Processing object: {obj.name}")
            set_vertex_colors_for_object(obj, cam, r=colors[count])
            count += 1
    

process_all_objects() 

