import torch
import cv2
import numpy as np

class GradCAMEngine:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register PyTorch hooks for feature activation and gradient capture
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Target raw logit score for backward pass
        score = output[0]
        score.backward(retain_graph=True)
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # Global Average Pooling of gradients to calculate channel weights
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

    def generate_overlay(self, input_tensor: torch.Tensor, original_img_np: np.ndarray):
        cam = self.generate_heatmap(input_tensor)
        
        # Resize heatmap to match original radiograph dimensions
        h, w = original_img_np.shape[:2]
        heatmap_resized = cv2.resize(cam, (w, h))
        
        # Apply JET colormap and convert BGR to RGB
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Blend original radiograph (60%) with heatmap overlay (40%)
        overlay = cv2.addWeighted(original_img_np, 0.6, heatmap_color, 0.4, 0)
        return heatmap_resized, overlay