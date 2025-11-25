import torch
import torch.nn as nn
from model.waft_a2_diffusion_in_time import WAFTv2_FlowDiffuser_TwoStage

class Args:
    def __init__(self):
        self.algorithm = 'waft-a2'
        self.feature_encoder = 'twins' # or 'dav2', 'dinov3'
        self.iterative_module = 'vits' # Assuming this is a valid key in MODEL_CONFIGS
        self.iters = 6
        self.mixed_precision = False
        self.var_max = 10.0
        self.var_min = -10.0
        self.image_size = [384, 512]

def test_model():
    args = Args()
    print("Initializing model...")
    device = torch.device('cpu')
    try:
        model = WAFTv2_FlowDiffuser_TwoStage(args)
        model.to(device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    B, C, H, W = 2, 3, 384, 512
    image1 = torch.randn(B, C, H, W).to(device) * 255
    image2 = torch.randn(B, C, H, W).to(device) * 255
    flow_gt = torch.randn(B, 2, H, W).to(device)

    print("Running forward pass (Training)...")
    model.train()
    try:
        output = model(image1, image2, flow_gt=flow_gt)
        print("Forward pass (Training) successful.")
        print("Output keys:", output.keys())
        if 'flow' in output:
            print("Flow output shape:", output['flow'][-1].shape)
    except Exception as e:
        print(f"Forward pass (Training) failed: {e}")
        import traceback
        traceback.print_exc()

    print("Running forward pass (Inference)...")
    model.eval()
    try:
        with torch.no_grad():
            output = model(image1, image2)
        print("Forward pass (Inference) successful.")
        if 'flow' in output:
            print("Flow output shape:", output['flow'][-1].shape)
    except Exception as e:
        print(f"Forward pass (Inference) failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
