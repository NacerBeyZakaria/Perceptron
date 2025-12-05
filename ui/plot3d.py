# ui/plot3d.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)


def _forward_output_only(mlp, X):
    """
    Call mlp.forward(X) and always return only the network output as a 2D array.
    This is robust to different forward signatures:
      - y = forward(X)
      - (a1, a2, y) = forward(X)
      - (activations, y) = forward(X)
    """
    res = mlp.forward(X)

    # If forward returns a tuple/list, take the last element as output
    if isinstance(res, (tuple, list)):
        out = res[-1]
    else:
        out = res

    out = np.asarray(out)

    # Ensure shape (N, 1) or (N,) becomes (N,)
    if out.ndim == 2 and out.shape[1] == 1:
        out = out[:, 0]

    return out


def show_mlp_3d_surface(mlp, X, resolution=60):
    """
    Draw a 3D decision surface for a trained MLP model.
    mlp.forward() must accept a batch of points and return outputs in [0,1]
    (or a tuple where the last element is the output).
    X: training inputs, shape (N, 2).
    """

    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("show_mlp_3d_surface expects X with shape (N, 2)")

    # define XY range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # flatten grid for forward pass
    grid = np.c_[xx.ravel(), yy.ravel()]

    # MLP nonlinear output surface (class probabilities or regression output)
    Z = _forward_output_only(mlp, grid)      # shape (resolution^2,)
    Z = Z.reshape(xx.shape)

    # 3D Plot
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')

    # surface
    surf = ax.plot_surface(xx, yy, Z,
                           cmap="coolwarm",
                           edgecolor='none',
                           alpha=0.85)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8)

    # data points projected with their MLP output
    for point in X:
        # forward expects batch; use point[None, :] to keep 2D
        out_pt = _forward_output_only(mlp, point[None, :])
        z_pt = float(out_pt.squeeze())
        ax.scatter(point[0], point[1], z_pt,
                   color='black', s=40)

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("MLP Output (ŷ)")
    ax.set_title("3D Nonlinear Decision Surface — MLP")

    plt.show()
