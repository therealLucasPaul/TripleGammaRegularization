# this is the base package for triple gamma regularizazion

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import log_hyperu as hyperu
import numpy as np
from sklearn.model_selection import train_test_split


class TripleGammaRegularization:
    def __init__(self, input_dim=1, output_dim=1, lr=0.01, bias=False):
        self.model = nn.Linear(input_dim, output_dim, bias=bias)  # Linear regression model
        self.mse = nn.MSELoss()  # MSE loss for regression
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)  # SGD optimizer

    def __str__(self):
        return f'Triple Gamma Model'
    
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, X, y, a:float, c:float, kappa:float, lamda:float, iter=1000):
        # Catch non valid inputs
        if lamda <0: 
            print(f'ERROR: Regularization Parameter Lambda has to be positive! A value of {lamda} has been chosen.')
            return None
        if a <= 0.5 or c <= 0:
            print(f'ERROR: Parameters for Triple Gamma Regularizazion out of range. See documentation!')
            return None
        if kappa == 0:
            print(f'Warning: Paramter kappa has been set to zero. The penalty will not have any influence on the optimization.')

        # FIT
        for epoch in range(iter):
            predictions = self.model(X)
            loss = self.mse(predictions, y)

            # Add custom penalty
            penalty = self.TGR_Penalty(self.model.weight, a, c, kappa)
            total_loss = loss + lamda*penalty

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{iter}, Loss: {total_loss.item():.4f}')

        print(f'Model trained!')            

    def predict(self, X):
        with torch.no_grad():
            return self.model(X)

    def get_parameters(self):
        return {
            "weights": self.model.weight.data.clone()
        }

    def showParameterStructure(self, a, c, kappa, lamda: float = 1, xrange:tuple = (-2,2)):
        x_vals = torch.linspace(xrange[0], xrange[1], int((xrange[1]-xrange[0])/0.01))
        span = torch.sign(x_vals) * torch.pow(torch.abs(x_vals), 0.5) * (xrange[1]-xrange[0]) # Make interval around zero more granular

        y_vals = list()
        for x in span:
            y_vals.append(self.TGR_Penalty(x, a, c, kappa).item())
        plt.plot(x_vals, y_vals, label=f'a = {a}\nc = {c}\n$\\kappa$={kappa}')

        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('TGR Penalty Value')
        plt.title(f'TGR Penalty Structure for \na = {a}, c = {c} and $\\kappa$ = {kappa}', fontsize=10)

        plt.legend()
        plt.grid(True)
        plt.show()

    def TGR_Penalty(self, x, a: float, c: float, kappa: float):
        # Compute Phi value for third parameter input of the hypergeometric function
        phi = (2*c)/((kappa**2)*a)
        parameter1 = torch.tensor([[c+0.5]])
        parameter2 = torch.tensor([[1.5-a]])
        if torch.is_tensor(x) == True:
            parameter3 = (x**2) / (2 * phi)
        else:
            parameter3 = torch.tensor([[(x**2) / (2 * phi)]])

        
        # Compute the penalty term using Triple Gamma Regularization
        penalty = torch.sum(-hyperu.log_hyperu(parameter1,parameter2,parameter3))
        
        return penalty
    

if __name__ == "__main__":
    # Sample dataset
    torch.manual_seed(42)

    # Generate synthetic dataset
    n_samples = 200
    n_features = 10

    # Features with multicollinearity and noise
    X = torch.rand(n_samples, n_features) * 10
    X[:, 1] = X[:, 0] + torch.randn(n_samples) * 0.1  # Add collinearity
    X[:, 2] = 0.5 * X[:, 0] + torch.randn(n_samples) * 0.1  # More collinearity
    y = 3 * X[:, 0] - 2 * X[:, 3] + 5 + torch.randn(n_samples) * 5  # Linear target with noise

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors (if using PyTorch exclusively)
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    # Initialize and train the model
    model = TripleGammaRegularization(input_dim=10, output_dim=1,lr=0.001, bias=False)
    model.fit(X_train, y_train, 0.75, 0.1, 2, 0)

    # Make predictions
    predictions = model.predict(X_test)
    XTX = torch.matmul(X_train.T, X_train)  # X^T X
    XTy = torch.matmul(X_train.T, y_train)  # X^T y
    ols_weights = torch.linalg.solve(XTX, XTy)  # Compute (X^T X)^-1 X^T y

    # Compute OLS predictions
    ols_predictions = torch.matmul(X_test, ols_weights)

    # Display learned parameters
    print(f'MSE Loss OLS: {np.square(np.subtract(y_test, ols_predictions)).mean()}')
    print(f'MSE Loss TGR: {np.square(np.subtract(y_test, predictions)).mean()}')
    
    #params = model.get_parameters()
    #print("Learned Parameters:", params)