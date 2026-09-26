
import numpy as np
import torch
import torch.nn as nn
 
 
class RecurrentNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1, context_mode="elman",
                 learn_decay=True, init_decay=0.5):
        super().__init__()
        assert context_mode in ("elman", "jordan", "multi")
        self.context_mode = context_mode
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if context_mode == "elman":
            context_dim = hidden_dim
        elif context_mode == "jordan":
            context_dim = output_dim
        else:
            context_dim = hidden_dim + output_dim
        self.context_dim = context_dim

        self.ih = nn.Linear(input_dim + context_dim, hidden_dim)
        self.ho = nn.Linear(hidden_dim, output_dim)

        init_logit = float(np.log(init_decay / (1 - init_decay)))
        if context_mode == "multi":
            self.alpha_h_logit = nn.Parameter(torch.full((hidden_dim,), init_logit),
                                               requires_grad=learn_decay)
            self.alpha_o_logit = nn.Parameter(torch.full((output_dim,), init_logit),
                                               requires_grad=learn_decay)
        else:
            self.alpha_logit = nn.Parameter(torch.full((context_dim,), init_logit),
                                             requires_grad=learn_decay)

    def forward(self, x):
        """x: (batch, timesteps, input_dim) -> final-timestep output (batch, output_dim)"""
        batch, T, _ = x.shape
        context = torch.zeros(batch, self.context_dim, device=x.device, dtype=x.dtype)
        o_t = None

        for t in range(T):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, context], dim=1)
            h_t = torch.tanh(self.ih(combined))
            o_t = self.ho(h_t)

            if self.context_mode == "elman":
                alpha = torch.sigmoid(self.alpha_logit)
                context = (1 - alpha) * context + alpha * h_t
            elif self.context_mode == "jordan":
                alpha = torch.sigmoid(self.alpha_logit)
                context = (1 - alpha) * context + alpha * o_t
            else:
                alpha_h = torch.sigmoid(self.alpha_h_logit)
                alpha_o = torch.sigmoid(self.alpha_o_logit)
                context_h = (1 - alpha_h) * context[:, :self.hidden_dim] + alpha_h * h_t
                context_o = (1 - alpha_o) * context[:, self.hidden_dim:] + alpha_o * o_t
                context = torch.cat([context_h, context_o], dim=1)

        return o_t
 
 
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64,
                 lr=0.01, patience=10, verbose=True, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = model.to(device)
 
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            f"train_model got an empty split (n_train={len(X_train)}, "
            f"n_val={len(X_val)}). This usually means the window size is "
            f"too large relative to this dataset/split's length -- try a "
            f"smaller window size or a larger split."
        )
 
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
 
    n = X_train_t.shape[0]
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
 
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_losses = []
 
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
 
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
 
        train_loss = float(np.mean(epoch_losses))
 
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
 
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
 
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
 
        if patience is not None:
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    model.load_state_dict(best_state)
                    break
 
    return history, model
 
 
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        pred = model(X_t)
        mse = torch.mean((pred - y_t) ** 2).item()
        rmse = mse ** 0.5
        mae = torch.mean(torch.abs(pred - y_t)).item()
    return {"mse": mse, "rmse": rmse, "mae": mae}