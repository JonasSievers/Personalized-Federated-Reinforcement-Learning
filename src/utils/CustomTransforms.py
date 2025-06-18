from torchrl.envs import Transform


'''
A custom observation standardization transform.
'''
class CustomObservationStandardization(Transform):
    def __init__(self):
        super().__init__()
        pass


'''
A custom action scaler transform.
'''
class CustomActionScaler(Transform):
    def __init__(self, action_spec):
        super().__init__(in_keys=[], out_keys=[], in_keys_inv=["action"], out_keys_inv=["action"])
        self._action_low = action_spec.low
        self._action_high = action_spec.high

    def _inv_apply_transform(self, action):
        return self._action_low + (self._action_high - self._action_low) * 0.5 * (action + 1)