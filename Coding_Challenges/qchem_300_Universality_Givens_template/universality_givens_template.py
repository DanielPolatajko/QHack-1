#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #

    dev = qml.device('default.qubit', wires=6)

    @qml.qnode(dev)
    def circuit(params):
        qml.BasisState(np.array([1, 1, 0, 0,0,0]), wires=range(6))
        qml.DoubleExcitation(params[0], wires=range(4))
        qml.DoubleExcitation(params[1], wires=range(2, 6))
        qml.ctrl(qml.SingleExcitation, control=0)(params[2], wires=[1, 3])
        return qml.state()

    def cost(params):
        out = circuit(params)
        return np.real((c-out[3]) ** 2 + (b-out[12]) ** 2 + (d-out[36]) ** 2 + (a - out[48]) ** 2)

    params = 0.01 * np.random.randn(3, requires_grad=True)

    opt = qml.optimize.NesterovMomentumOptimizer(0.5)

    epochs = 500

    # QHACK #

    for i in range(epochs):
        params, _ = opt.step_and_cost(cost, params)

    return params

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")
