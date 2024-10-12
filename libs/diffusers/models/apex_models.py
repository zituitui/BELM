from typing import Optional
from apex import fused_dense

class LoRACompatibleFusedLinear(fused_dense.FusedDense):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer):
        self.lora_layer = lora_layer

    def forward(self, x):
        if x.ndim == 2:
            y = super().forward(x)
        elif x.ndim == 3:
            n, l, c = x.size()
            y = super().forward(x.view(n * l, c)).view(n, l, -1)
        elif x.ndim == 4:
            n, l1, l2, c = x.size()
            y = super().forward(x.view(n * l1 * l2, c)).view(n, l1, l2, -1)

        if not self.lora_layer is None:
            y = y +  self.lora_layer(x)
        return y

class FusedLinear(fused_dense.FusedDense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if x.ndim == 2:
            return super().forward(x)
        elif x.ndim == 3:
            n, l, c = x.size()
            return super().forward(x.view(n * l, c)).view(n, l, -1)
        elif x.ndim == 4:
            n, l1, l2, c = x.size()
            return super().forward(x.view(n * l1 * l2, c)).view(n, l1, l2, -1)