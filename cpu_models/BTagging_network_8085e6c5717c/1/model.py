import json
import os
from pathlib import Path

import onnxruntime as ort
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the ONNX session. 'args' contains the model path.
        """
        self.model_config = json.loads(args["model_config"])

        # Determine the path to the model file relative to this script
        model_path = os.path.join(args['model_repository'], args['model_version'], 'model.onnx')

        # Set execution provider (CPU in this case based on your config)
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # Pre-fetch output names from ONNX to ensure correct mapping
        self.output_names = [output.name for output in self.session.get_outputs()]

        parameters = self.model_config["parameters"]
        def get_parameter(name):
            if name not in parameters:
                raise ValueError(f"Parameter {name} is required but not provided.")
            return parameters[name]["string_value"]

        self.save_data_to_json = get_parameter("save_inputs_to_json").lower() == "true"
        if self.save_data_to_json:
            self.save_json_dir = Path("input_json") / "DML"
            self.save_json_dir.mkdir(parents=True, exist_ok=True)
            self.counter = 0

    def execute(self, requests):
        """
        Main execution logic for Triton requests.
        """
        responses = []

        for request in requests:
            # 1. Gather all inputs defined in your config
            input_names = ["jet_features", "track_features", "flow_features", "electron_features"]
            onnx_inputs = {}
            json_data = {}

            for name in input_names:
                tensor = pb_utils.get_input_tensor_by_name(request, name)
                if tensor is not None:
                    onnx_inputs[name] = tensor.as_numpy()
                    if self.save_data_to_json:
                        json_data[name] = {
                            "content": onnx_inputs[name].tolist(),
                            "shape": onnx_inputs[name].shape
                        }
            if self.save_data_to_json:
                save_path = self.save_json_dir / f"request_{self.counter}.json"
                with open(save_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                self.counter += 1

            # 2. Run Inference
            onnx_outputs = self.session.run(self.output_names, onnx_inputs)

            # 3. Map ONNX outputs back to Triton tensors
            output_tensors = []
            for name, data in zip(self.output_names, onnx_outputs):
                # We use pb_utils.Tensor to wrap the numpy array
                out_tensor = pb_utils.Tensor(name, data)
                output_tensors.append(out_tensor)

            # 4. Create response object
            inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(inference_response)

        return responses

    def finalize(self):
        """
        Cleanup on model unload.
        """
        del self.session
