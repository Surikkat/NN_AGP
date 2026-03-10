import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_width=128, hidden_layers=2, output_dim=1):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_width))
            layers.append(nn.ReLU())
            dim = hidden_width
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class NN_UCB:
    def __init__(self,
                 context_dim,
                 hidden_width=256,
                 hidden_layers=2,
                 lambda_=1.0,
                 beta=1.0,
                 J=10,
                 lr=1e-3,
                 device=None,
                 seed=0):
        self.context_dim=context_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.net = SimpleMLP(input_dim=context_dim,
                             hidden_width=hidden_width,
                             hidden_layers=hidden_layers).to(self.device)
        
        self.param_shapes = [p.shape for p in self.net.parameters()]
        self.num_params = sum(p.numel() for p in self.net.parameters())

        self.lambda_ = float(lambda_)
        self.A_diag = (self.lambda_) * np.ones(self.num_params, dtype=float)

        self.buffer_X = []
        self.buffer_y = []

        self.J = int(J)
        self.lr = float(lr)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)

        self.beta = float(beta)

        self.eps = 1e-8

    def _params_to_vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self.net.parameters()], dim=0)
    
    def _grad_feature(self, x):
        self.net.zero_grad()
        if not isinstance(x, torch.Tensor):
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            x_t = x.to(self.device).unsqueeze(0)

        out = self.net(x_t)
        out.backward()
        grads = []
        for p in self.net.parameters():
            g = p.grad
            if g is None:
                grads.append(torch.zeros(p.numel(), device=self.device))
            else:
                grads.append(g.detach().reshape(-1))
        phi = torch.cat(grads, dim=0).cpu().numpy()
        return phi
    
    def _update_A_inv_sherman_morrison(self, phi):
        phi = phi.reshape(-1, 1)
        Ainv_phi = self.A_inv.dot(phi)
        denom = float(1.0 + (phi.T.dot(Ainv_phi)))
        if abs(denom) < self.eps:
            denom = self.eps
        numer = Ainv_phi.dot((Ainv_phi.T))
        self.A_inv = self.A_inv - (numer / denom)
    
    def select_action(self, contexts):
        if isinstance(contexts, np.ndarray):
            ctxs = contexts
        else:
            ctxs = np.array(contexts)
        
        k = ctxs.shape[0]
        ucbs = []
        self.net.eval()


        for i in range(k):
            x_np = ctxs[i]
            with torch.no_grad():
                x_t = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                mu = self.net(x_t).item()
            phi = self._grad_feature(x_np)
            denom = self.A_diag
            v = float(np.sqrt(np.sum((phi ** 2) / (denom + self.eps))))
            ucb = mu + self.beta * v
            ucbs.append(ucb)
        return int(np.argmax(ucbs))

    def update(self, context, action, reward):
        if isinstance(context, torch.Tensor):
            ctx = context.cpu()
        else:
            ctx = np.array(context, dtype=np.float32)
        self.buffer_X.append(ctx)
        self.buffer_y.append(float(reward))

        if len(self.buffer_X) == 0:
            return

        self.net.train()
        X = torch.tensor(np.stack([x if isinstance(x, np.ndarray) else x.numpy() for x in self.buffer_X]),
                         dtype=torch.float32, device=self.device)
        y = torch.tensor(self.buffer_y, dtype=torch.float32, device=self.device)
        loss_fn = nn.MSELoss()

        for _ in range(self.J):
            self.optimizer.zero_grad()
            preds = self.net(X)
            loss = loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()

        phi = self._grad_feature(context)
        self.A_diag += (phi ** 2)