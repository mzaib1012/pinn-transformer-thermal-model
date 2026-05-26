import torch

def compute_physics_loss(model, collocation_points, alpha=0.01, gamma=0.5):
    """
    Computes the residual loss of the 1D Heat Equation.
    
    collocation_points: Tensor of shape (N, 4) containing:
                        [time (t), height (x), ambient_temp (T_amb), load_current (I)]
    alpha: Thermal diffusivity constant
    gamma: Heat generation coefficient from load current losses (I^2 * R)
    """
    # Clone and enable gradient tracking on inputs to allow automatic differentiation
    points = collocation_points.clone().detach().requires_grad_(True)
    
    t = points[:, 0:1]
    x = points[:, 1:2]
    T_amb = points[:, 2:3]
    I = points[:, 3:4]
    
    # Recombine inputs to pass through the model
    input_tensor = torch.cat([t, x, T_amb, I], dim=1)
    T_pred = model(input_tensor)
    
    # 1. First derivative of Temperature with respect to Time (dT/dt)
    dT_dt = torch.autograd.grad(
        T_pred, t, 
        grad_outputs=torch.ones_like(T_pred),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # 2. First derivative of Temperature with respect to Space/Height (dT/dx)
    dT_dx = torch.autograd.grad(
        T_pred, x, 
        grad_outputs=torch.ones_like(T_pred),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # 3. Second derivative of Temperature with respect to Space/Height (d^2T/dx^2)
    dT_dx2 = torch.autograd.grad(
        dT_dx, x, 
        grad_outputs=torch.ones_like(dT_dx),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # 4. Define the Heat Equation PDE residual:
    # PDE: dT/dt - alpha * d^2T/dx^2 - gamma * I^2 = 0
    # In reality, the residual should be as close to zero as possible.
    pde_residual = dT_dt - alpha * dT_dx2 - gamma * (I ** 2)
    
    # 5. Compute mean squared error of the physical mismatch
    physics_loss = torch.mean(pde_residual ** 2)
    
    return physics_loss
