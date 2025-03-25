import numpy as np
from scipy.optimize import minimize

from dataclasses import dataclass
from typing import AnyStr, Callable


@dataclass
class Controller:
    constr_method: AnyStr
    model: Callable[[np.array, np.array], np.array]
    n_states: int
    n_inputs: int
    n_pred: int
    t_max: int
    Q: np.array
    R: np.array
    term_constr: bool
    state_bounds: np.array
    minimize_method: AnyStr = "SLSQP"
    verbose: bool = True

    def __post_init__(self):
        self.dt = self.t_max / self.n_pred
        self.max_state = np.max(self.state_bounds)
        self.min_state = np.min(self.state_bounds)

    def euler_step(self, x: np.array, u: np.array):
        return x + self.dt * self.model(x, u)

    def RK4_step(self, x: np.array, u: np.array):
        k1 = self.model(x, u)
        k2 = self.model(x + k1 * self.dt / 2, u)
        k3 = self.model(x + k2 * self.dt / 2, u)
        k4 = self.model(x + k3 * self.dt, u)

        return x + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def shooting_step(self, x, u):
        """Simulate one shooting interval"""
        return self.RK4_step(x, u)

    def objective(self, X: np.array, x_ref: np.array):
        """Objective function: minimize control effort"""
        X = X.reshape([self.n_pred, self.n_states + self.n_inputs], order="F")
        X_state = X[:, : self.n_states]
        x_ref = np.tile(x_ref, (len(X_state), 1))
        U = X[:, -self.n_inputs :]

        state_cost = np.sum(
            [
                ((X_state[i]) @ self.Q) @ (X_state[i]).T
                for i in range(len(X_state))
            ]
        )
        input_cost = np.sum([(U[i] @ self.R) @ U[i] for i in range(len(U))])

        return input_cost + state_cost

    def constraints(self, X: np.array, x0: np.array, x_ref: np.array):
        """Constraints for direct multiple shooting"""
        X = X.reshape([self.n_pred, self.n_states + self.n_inputs], order="F")
        t = np.arange(0, self.n_pred, self.dt)
        ceq = []

        if self.constr_method == "DMS":
            for i in range(self.n_pred - 1):
                X_next = self.shooting_step(
                    X[i, : self.n_states], X[i, -self.n_inputs :]
                )
                ceq.extend((X[i+1, : self.n_states] - X_next).tolist())
        elif self.constr_method == "COLL":
            for i in range(self.n_pred - 1):
                ceq.extend(
                    X[i + 1, : self.n_states]
                    - X[i, : self.n_states]
                    - (self.dt / 2)
                    * (
                        self.model(X[i, : self.n_states], X[i, -self.n_inputs :])
                        + self.model(
                            X[i + 1, : self.n_states], X[i + 1, -self.n_inputs :]
                        )
                    )
                )

        ceq.extend(X[0, : self.n_states] - x0.tolist())

        if self.term_constr:
            ceq.extend(X[-1, : self.n_states] - x_ref.tolist())
        return ceq

    def ineq_constraints(self, X: np.array):
        X = X.reshape([self.n_pred, self.n_states + self.n_inputs], order="F")
        x = X[:, : self.n_states]
        u = X[:, -self.n_inputs :]

        ineq = []
        for state in range(self.n_states):
            for idx in range(self.n_pred):
                ineq.extend([self.max_state - x[idx, state]])
                ineq.extend([x[idx, state] - self.min_state])

        for idx in range(self.n_pred):
            ineq.extend([u[idx, 1] - 10])  # due to the ship model, u[1] should always be > 1
        return ineq

    def optimize(self, x0: np.array, xref: np.array):
        optimizer_dict = {"x": np.array, "u": np.array, "flag": False, "cost": float}

        X_guess = np.full([self.n_pred, self.n_states + self.n_inputs], np.nan)
        for i in range(self.n_states + self.n_inputs):
            if i < self.n_states:
                X_guess[:, i] = np.linspace(0.1, 1, self.n_pred)

        X_guess[:, self.n_states] = 0.5 * np.ones(self.n_pred)  # delta command
        X_guess[:, self.n_states+1] = 10 * np.ones(self.n_pred)

        X_guess = X_guess.flatten(order="F")
        new_constraints = lambda x: self.constraints(x, x0, xref)
        new_objective = lambda x: self.objective(x, xref)
        result = minimize(
            new_objective,
            X_guess,
            method=self.minimize_method,
            constraints=[
                {"type": "eq", "fun": new_constraints},
                {"type": "ineq", "fun": self.ineq_constraints},
            ],
        )

        res_reshaped = result.x.reshape(
            [self.n_pred, self.n_states + self.n_inputs], order="F"
        )
        optimizer_dict["x"] = res_reshaped[:, : self.n_states]
        optimizer_dict["u"] = res_reshaped[:, -self.n_inputs :]
        optimizer_dict["flag"] = result.success
        optimizer_dict["cost"] = result.fun

        return optimizer_dict
