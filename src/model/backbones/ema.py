import copy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model).eval()  # Create a shadow model
        self.decay = decay
        self.model = model
        self._backup = {}  # Backup of the model's parameters

    def update(self):
        with torch.no_grad():
            msd = self.model.state_dict()  # Model state_dict
            esd = self.ema_model.state_dict()  # EMA state_dict
            for name, param in msd.items():
                if name in esd:
                    ema_param = esd[name]
                    ema_param.copy_(ema_param * self.decay + param * (1 - self.decay))

    def apply_shadow(self):
        """Use EMA weights for the model"""
        self._backup = self.model.state_dict()
        self.model.load_state_dict(self.ema_model.state_dict())

    def restore(self):
        """Restore the original weights of the model"""
        self.model.load_state_dict(self._backup)
