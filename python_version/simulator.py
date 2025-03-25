import time
import numpy as np
from typing import AnyStr, Callable

try:
    from tqdm import trange
except ImportError as e:
    trange = range
    print(f"[WARNING] No progress bar will be displayed, install tqdm")


class Simulator:
    def __init__(
        self,
        controller,
        x_0: np.array,
        x_r: np.array,
        sim_steps: int,
        plant: Callable[[np.array, np.array, float], np.array],
        input_bounds: np.array = None,
    ):
        assert len(x_0) == len(x_r)

        self.controller = controller
        self.x0 = x_0
        self.x_ref = x_r
        self.sim_steps = sim_steps
        self.plant = plant
        self.input_bounds = input_bounds

    def get_orbit(self):
        # Initialize states [x] and inputs [u] matrices with np.nan values
        x = np.full([self.sim_steps + 1, self.controller.n_states], np.nan)
        u = np.full([self.sim_steps + 1, self.controller.n_inputs], np.nan)
        e = np.full([self.sim_steps,                        1], np.nan)

        # Populate the initial state, input in x, u at index [0]
        x[0] = self.x0
        u[0] = self.controller.optimize(self.x0, self.x_ref)["u"][0]
        for k in trange(self.sim_steps):
            opt_res = self.controller.optimize(x[k], self.x_ref)

            if opt_res["flag"] == "False":
                exit
                
            x[k + 1] = self.plant(x[k], u[k])
            u[k + 1] = opt_res["u"][0]
            e[k] = opt_res["cost"]

        return x, u, e
