import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np

def main():

    onnx_path = "models/both"
    model = onnx.load(onnx_path + "/Pyramid.onnx")

    # --- version_number as float32, shape [1], value [3.0] ---
    if not any(init.name == "version_number.1" for init in model.graph.initializer):
        version_number_tensor = numpy_helper.from_array(np.array([3.0], dtype=np.float32), name="version_number.1")
        model.graph.initializer.append(version_number_tensor)
    version_number_identity = helper.make_node(
        'Identity',
        inputs=["version_number.1"],
        outputs=["version_number"],
        name="Identity_version_number"
    )
    model.graph.node.append(version_number_identity)
    if not any(output.name == "version_number" for output in model.graph.output):
        version_number_output = helper.make_tensor_value_info("version_number", TensorProto.FLOAT, [1])
        model.graph.output.append(version_number_output)

    # --- memory_size_vector as float32, shape [1], value [0.0] ---
    if not any(init.name == "memory_size_vector" for init in model.graph.initializer):
        memory_size_vector_tensor = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="memory_size_vector")
        model.graph.initializer.append(memory_size_vector_tensor)

    # --- discrete_act_size_vector as float32, shape [1,1], value [[5.0]] ---
    if not any(init.name == "discrete_act_size_vector" for init in model.graph.initializer):
        discrete_act_size_vector_tensor = numpy_helper.from_array(np.array([[5.0]], dtype=np.float32), name="discrete_act_size_vector")
        model.graph.initializer.append(discrete_act_size_vector_tensor)

    # --- memory_size as float32, shape [1], value [0.0] ---
    if not any(init.name == "memory_size.1" for init in model.graph.initializer):
        memory_size_tensor = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="memory_size.1")
        model.graph.initializer.append(memory_size_tensor)
    memory_size_identity = helper.make_node(
        'Identity',
        inputs=["memory_size.1"],
        outputs=["memory_size"],
        name="Identity_memory_size"
    )
    model.graph.node.append(memory_size_identity)
    if not any(output.name == "memory_size" for output in model.graph.output):
        memory_size_output = helper.make_tensor_value_info("memory_size", TensorProto.FLOAT, [1])
        model.graph.output.append(memory_size_output)

    # --- discrete_action_output_shape as float32, shape [1,1], value [[5.0]] ---
    if not any(init.name == "discrete_action_output_shape.1" for init in model.graph.initializer):
        discrete_action_output_shape_tensor = numpy_helper.from_array(np.array([[5.0]], dtype=np.float32), name="discrete_action_output_shape.1")
        model.graph.initializer.append(discrete_action_output_shape_tensor)
    discrete_action_output_shape_identity = helper.make_node(
        'Identity',
        inputs=["discrete_action_output_shape.1"],
        outputs=["discrete_action_output_shape"],
        name="Identity_discrete_action_output_shape"
    )
    model.graph.node.append(discrete_action_output_shape_identity)
    if not any(output.name == "discrete_action_output_shape" for output in model.graph.output):
        discrete_action_output_shape_output = helper.make_tensor_value_info("discrete_action_output_shape", TensorProto.FLOAT, [1, 1])
        model.graph.output.append(discrete_action_output_shape_output)

    # Save the modified model
    onnx.save(model, onnx_path + "/Pyramid_modified.onnx")
    print("Added version_number.1, memory_size_vector, and discrete_act_size_vector as float32 to ONNX model.")
    
if __name__ == "__main__":
    main()
    print("ONNX model modification complete.")