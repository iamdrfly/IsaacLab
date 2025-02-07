import torch

# Funzione per calcolare il quaternione che rappresenta l'orientamento di un vettore
def vector_to_quaternion(X, device):
    # Calcola la norma del vettore X
    norm_X = torch.norm(X, dim=1)

    # Calcola l'angolo di rotazione theta
    theta = torch.acos(X[:, 0] / norm_X)

    # Calcola l'asse di rotazione come il prodotto vettoriale tra (1, 0, 0) e X
    w_x = torch.ones_like(X, device=device)  # Il vettore di riferimento (1, 0, 0)
    w_x = w_x * torch.tensor([1, 0, 0], device=device)
    axis_of_rotation = torch.cross(w_x, X, dim=1)

    # Calcola la norma dell'asse di rotazione
    norm_axis = torch.norm(axis_of_rotation, dim=1)

    # Normalizza l'asse di rotazione
    axis_of_rotation_normalized = axis_of_rotation / norm_axis.unsqueeze(1)

    # Calcola il quaternione
    qw = torch.cos(theta / 2)
    qx = axis_of_rotation_normalized[:, 0] * torch.sin(theta / 2)
    qy = axis_of_rotation_normalized[:, 1] * torch.sin(theta / 2)
    qz = axis_of_rotation_normalized[:, 2] * torch.sin(theta / 2)

    # Il quaternione è: q = qw + qx*i + qy*j + qz*k
    quaternion = torch.stack([qw, qx, qy, qz], dim=1)

    return quaternion