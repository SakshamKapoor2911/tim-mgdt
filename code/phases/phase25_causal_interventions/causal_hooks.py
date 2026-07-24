import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes


def compute_procrustes_rotation(
    teacher_hidden: torch.Tensor, student_hidden: torch.Tensor
) -> torch.Tensor:
    teacher_np = (
        teacher_hidden.detach().float().reshape(-1, teacher_hidden.shape[-1]).cpu().numpy()
    )
    student_np = (
        student_hidden.detach().float().reshape(-1, student_hidden.shape[-1]).cpu().numpy()
    )
    R, _ = orthogonal_procrustes(student_np, teacher_np)
    return torch.from_numpy(R).float()


def random_orthogonal_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(dim, dim))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return torch.from_numpy(Q).float()


def _get_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    return output


class HiddenStateCollector:
    def __init__(self):
        self.hidden = None

    def __call__(self, module, input, output):
        h = _get_hidden(output).detach().cpu()
        self.hidden = h.reshape(-1, h.shape[-1])
        return output


class SubspaceInterventionHook:
    def __init__(self):
        self.R = None

    def set_rotation(self, R: torch.Tensor):
        self.R = R

    def __call__(self, module, input, output):
        if self.R is None:
            return output
        h = _get_hidden(output)
        R_dev = self.R.to(h.device, dtype=h.dtype)
        h_intervened = torch.matmul(h, R_dev)
        if isinstance(output, tuple):
            return (h_intervened,) + output[1:]
        return h_intervened


def register_collector(model, layer_idx: int):
    collector = HiddenStateCollector()
    handle = model.model.layers[layer_idx].register_forward_hook(collector)
    return collector, handle


def register_intervention(model, layer_idx: int):
    hook = SubspaceInterventionHook()
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    return hook, handle
