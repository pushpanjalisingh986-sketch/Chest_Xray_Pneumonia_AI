import cv2
import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        score = output[0, 0]
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

def overlay_cam_on_image(img_np: np.ndarray, cam_map: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w, _ = img_np.shape
    cam_resized = cv2.resize(cam_map, (w, h))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    blended = alpha * heatmap.astype(np.float32) + (1 - alpha) * img_np.astype(np.float32)
    return np.uint8(np.clip(blended, 0, 255))