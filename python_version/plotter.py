from cProfile import label
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Plotter:
    name: str
    n_states: int
    n_inputs: int
    n_errors: int
    methods: list
    save_plots: bool

    def plot_for_method(self, x: np.array, axes, lbl):
        for i in range(x.shape[1]):
            state_lbl = f"{lbl}[{i}]"

            if x.shape[1] == 1:
                axes.plot(self.timestep, x[:, i], label=state_lbl)
                axes.set_ylabel(state_lbl, rotation=45)
                axes.grid()
            else:
                axes[i].plot(self.timestep, x[:, i], label=state_lbl)
                axes[i].set_ylabel(state_lbl, rotation=45)
                axes[i].grid()

    def plot_state(self, x: np.array) -> None:
        """
        Assuming state array shape is [timestep_idx, ith_state]
        """

        self.timestep = np.arange(0, len(x)) 

        xi = list(np.hsplit(x, len(self.methods))) 

        state_fig, state_axs = plt.subplots(self.n_states, figsize=(10, 10))
        state_fig.suptitle(f"State plot for {self.name} constraint method")
        state_fig.subplots_adjust(hspace=0.4) 

        for i in range(len(xi)):
            self.plot_for_method(xi[i], state_axs, "x")
        
        state_axs[-1].set_xlabel("timestep")
        state_fig.legend(self.methods)
        if self.save_plots:
            if not os.path.exists("./plots"):
                os.mkdir("./plots")
            state_fig.savefig("./plots/state_plot.png")


    def plot_input(self, u: np.array) -> None:
        """
        Assuming input array shape is [timestep_idx, ith_input]
        """

        self.timestep = np.arange(0, len(u))

        ui = list(np.hsplit(u, len(self.methods))) 

        input_fig, input_axs = plt.subplots(self.n_inputs, figsize=(10, 5))
        input_fig.suptitle(f"Input plot for {self.name} constraint method")
        input_fig.subplots_adjust(hspace=0.5) 

        for i in range(len(ui)):
            self.plot_for_method(ui[i], input_axs, "u")

        input_axs[-1].set_xlabel("timestep")
        input_fig.legend(self.methods)
        if not os.path.exists("./plots"):
            os.mkdir("./plots")
        input_fig.savefig("./plots/input_plot.png")
    
    def plot_error(self, e: np.array):
        """
        Assuming input array shape is [timestep_idx, ith_error]
        """

        self.timestep = np.arange(0, len(e))
        
        ei = list(np.hsplit(e, len(self.methods))) 

        error_fig, error_axs = plt.subplots(self.n_errors, figsize=(10, 5))
        error_fig.suptitle(f"Error plot for {self.name} constraint method")
        error_fig.subplots_adjust(hspace=0.5) 

        for i in range(len(ei)):
            self.plot_for_method(ei[i], error_axs, "e")

        if not os.path.exists("./plots"):
            os.mkdir("./plots")
        error_fig.savefig("./plots/error_plot.png")