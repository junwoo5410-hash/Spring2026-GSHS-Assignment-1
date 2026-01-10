# Assignment 1: Gradient-Descent-Based Solver

## Objective
Write a Python program to implement a gradient descent solver that minimizes a function of the form:
```f(x, y) = x^2 + y^2 + 2x + 4y + 5```

## Instructions
1. Complete the `gradient_descent` and `gradient_f` functions in `gradient_descent.py`.
2. The `gradient_descent` function should:
   - Initialize `x`, `y` with `starting_values`.
   - Iteratively update `x`, `y` using the gradient descent update rule for `num_iterations` iterations.
   - Return the tuple `(x, y)` after `num_iterations` updates.
3. The `gradient_f` function should:
   - Return the tuple `(∂f/∂x, ∂f/∂y)` for the above function `f(x, y)` and coordinates `(x, y)`.

## Gradient Descent update rule
The update rule for gradient descent is:
```
    x_new = x - alpha * ∂f/∂x
    y_new = y - alpha * ∂f/∂y
```
Where:
- `∂f/∂x` is the partial derivative of `f(x, y)`.
- `∂f/∂y` is the partial derivative of `f(x, y)`.
- `alpha` is the learning rate.
