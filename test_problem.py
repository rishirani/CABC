import cvxpy as cp

# Create variables
x = cp.Variable()
y = cp.Variable()

# Introduce a new variable t for the reciprocal term
t = cp.Variable()

# Define constraints to make the original expression DCP compliant
constraints = [
    t * (y - x) == 1,  # t is the reciprocal of (y - x)
    y - x >= 0,        # Denominator must be positive for the division
    x >= 0             # For the expression to be well-defined
]

# Example: If the objective was to maximize x / (y-x)
# You would then write:
objective = cp.Maximize(x * t)
problem = cp.Problem(objective, constraints)
problem.solve()

# To verify the expression:
print(f"Is the new expression DCP compliant: {objective.is_dcp()}")