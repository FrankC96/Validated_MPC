import warnings
import numpy as np

from controller import Controller
from simulator import Simulator
from plotter import Plotter
from functools import partial

warnings.simplefilter("error", RuntimeWarning)

def model(x: np.array, u_c: np.array) -> np.array:
    """
    A mathematical model representing the plant, by a set of 1st order nonlinear equations.

    The nonlinear system is normalized by
        -> [L] distance between each wheel
        -> [V] maximum linear velocity of the robot
    """

    #  Normalization variables
    L   =  304.8          # Length of ship [m]
    g   =  9.8            # Acceleration of gravity [m/s**2]

    # Dimensional states and inputs
    u     = x[0]    
    v     = x[1] 
    r     = x[2]
    xpos  = x[3]
    ypos  = x[4]
    psi   = x[5] 
    U     = np.sqrt(x[1]**2 + x[2]**2)

    delta_c = u_c[0]  
    n_c     = u_c[1] / 60  # rps
    h_c     = 150 # assuming constant sea depth, not an input

    t   =  0.22
    Tm  =  50
    T   =  18.46

    cun =  0.605  
    cnn =  38.2

    Tuu = -0.00695
    Tun = -0.00063
    Tnn =  0.0000354

    m11 =  1.050         # 1 - Xudot
    m22 =  2.020         # 1 - Yvdot
    m33 =  0.1232        # kz**2 - Nrdot

    d11 =  2.020         # 1 + Xvr
    d22 = -0.752         # Yur - 1
    d33 = -0.231         # Nur - xG 

    Xuuz   = -0.0061;   YT     =  0.04;   NT      = -0.02
    Xuu    = -0.0377;   Yvv    = -2.400;  Nvr     = -0.300
    Xvv    =  0.3;      Yuv    = -1.205;  Nuv     = -0.451   
    Xudotz = -0.05;    Yvdotz = -0.387;  Nrdotz  = -0.0045
    Xuuz   = -0.0061;   Yurz   =  0.182;  Nurz    = -0.047
    Xvrz   =  0.387;    Yvvz   = -1.5;    Nvrz    = -0.120
    Xccdd  = -0.093;    Yuvz   =  0;      Nuvz    = -0.241
    Xccbd  =  0.152;    Yccd   =  0.208;  Nccd    = -0.098
    Xvvzz  =  0.0125;   Yccbbd = -2.16;   Nccbbd  =  0.688
    Yccbbdz= -0.191;  Nccbbdz =  0.344


    z = T / (h_c - T)
    beta = np.arctan(v / u)
    gT   = (1/L*Tuu*u**2 + Tun*u*n_c + L*Tnn*abs(n_c)*n_c)
    c    = np.sqrt(cun*u*n_c + cnn*n_c**2)

    gX   = 1/L*(Xuu*u**2 + L*d11*v*r + Xvv*v**2 + Xccdd*abs(c)*c*delta_c**2 \
        + Xccbd*abs(c)*c*beta*delta_c + L*gT*(1-t) + Xuuz*u**2*z \
        + L*Xvrz*v*r*z + Xvvzz*v**2*z**2)

    gY   = 1/L*(Yuv*u*v + Yvv*abs(v)*v + Yccd*abs(c)*c*delta_c + L*d22*u*r
        + Yccbbd*abs(c)*c*abs(beta)*beta*abs(delta_c) + YT*gT*L\
        + L*Yurz*u*r*z + Yuvz*u*v*z + Yvvz*abs(v)*v*z \
        + Yccbbdz*abs(c)*c*abs(beta)*beta*abs(delta_c)*z)     

    gLN  = Nuv*u*v + L*Nvr*abs(v)*r + Nccd*abs(c)*c*delta_c +L*d33*u*r\
        + Nccbbd*abs(c)*c*abs(beta)*beta*abs(delta_c) + L*NT*gT\
        + L*Nurz*u*r*z + Nuvz*u*v*z + L*Nvrz*abs(v)*r*z\
        + Nccbbdz*abs(c)*c*abs(beta)*beta*abs(delta_c)*z

    m11 = (m11 - Xudotz*z)
    m22 = (m22 - Yvdotz*z)
    m33 = (m33 - Nrdotz*z)

    # Dimensional state derivative
    xdot = np.array([
        gX/m11,
        gY/m22,
        gLN/(L**2*m33),
        np.cos(psi) * u - np.sin(psi) * v,
        np.sin(psi) * u + np.cos(psi) * v,
        r,
    ])

    return xdot

def plant(x: np.array, u: np.array, mag: float) -> np.array:
    """
    Adding random uniform noise to the dynamics to approximate the mismatch
    between the dynamics and the actual plant.
    """
    return model(x, u) + np.random.uniform(-mag, mag, (len(x),))


if __name__ == "__main__":
    x_init = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_ref = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 0.0])
    # x_ref = np.zeros(len(x_init))

    T_MAX = 1
    N_PRED = 30  # for 1 shooting / knot node both methods are identical
    N_SIM = 10
    N_STATES = len(x_init)
    N_INPUTS = 2

    PLANT_NOISE = 0.0

    STATE_BOUNDS = np.array([-1000, 1000])
    INPUT_BOUNDS = np.array([-1000, 1000])

    Q = 1 * np.eye(N_STATES)
    R = 1 * np.eye(N_INPUTS)

    # using the dynamics function to perform a prediction step
    mpc_coll = Controller(
        constr_method="COLL",
        model=model,
        n_states=N_STATES,
        n_inputs=N_INPUTS,
        n_pred=N_PRED,
        t_max=T_MAX,
        Q=Q,
        R=R,
        state_bounds=STATE_BOUNDS,
        minimize_method="SLSQP",
        term_constr=False,
    )
    mpc_dms = Controller(
        constr_method="DMS",
        model=model,
        n_states=N_STATES,
        n_inputs=N_INPUTS,
        n_pred=N_PRED,
        t_max=T_MAX,
        Q=Q,
        R=R,
        state_bounds=STATE_BOUNDS,
        minimize_method="SLSQP",
        term_constr=False,
    )

    # using the plant function to perform a simulation step
    sim_coll = Simulator(
        controller=mpc_coll,
        x_0=x_init,
        x_r=x_ref,
        sim_steps=N_SIM,
        plant=partial(plant, mag=PLANT_NOISE),
        input_bounds=None,
    )

    sim_dms = Simulator(
        controller=mpc_dms,
        x_0=x_init,
        x_r=x_ref,
        sim_steps=N_SIM,
        plant=partial(plant, mag=PLANT_NOISE),
        input_bounds=None,
    )

    x_coll, u_coll, e_coll = sim_coll.get_orbit()
    print(f"[{mpc_coll.constr_method}] Final x, u {x_coll[-1, :]} | {u_coll[-1, :]}")

    x_dms, u_dms, e_dms = sim_dms.get_orbit()
    print(f"[{mpc_dms.constr_method}] Final x, u {x_dms[-1, :]} | {u_dms[-1, :]}")
        
    plotter = Plotter(
        name=mpc_coll.constr_method,
        n_states=mpc_coll.n_states,
        n_inputs=mpc_coll.n_inputs,
        n_errors=1,
        methods=["coll", "dms"],
        save_plots=True
    )
    
    # FIXME: not showing grid, called on each ax object
    plotter.plot_state(np.hstack([x_coll, x_dms]))
    plotter.plot_input(np.hstack([u_coll, u_dms]))
    plotter.plot_error(np.hstack([e_coll, e_dms]))

