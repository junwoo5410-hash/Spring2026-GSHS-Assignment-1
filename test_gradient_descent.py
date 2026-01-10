import math
from gradient_descent import gradient_descent

def test_gradient_descent_1():
    '''
    Test Case: 1: f(x,y)=x2+y2+2x+4y+5
    '''
    result_x, result_y = gradient_descent((0, 0), 0.1, 100)
    expected_x, expected_y = (-0.9999, -1.9999)
    assert math.isclose(result_x, expected_x, abs_tol=1e-3)
    assert math.isclose(result_y, expected_y, abs_tol=1e-3)

def test_gradient_descent_2():
    '''
    Test Case: 2: f(x,y)=x2+y2+2x+4y+5
    Learning Rate: 0.5
    Starting Point: (5, 5)
    '''
    result_x, result_y = gradient_descent((5, 5), 0.5, 100)
    expected_x, expected_y = (-1, -2)
    assert math.isclose(result_x, expected_x, abs_tol=1e-3)
    assert math.isclose(result_y, expected_y, abs_tol=1e-3)

def test_gradient_descent_3():
    '''
    Test Case: 3: f(x,y)=x2+y2+2x+4y+5
    Num of Iterations: 50
    '''
    result_x, result_y = gradient_descent((0, 0), 0.1, 50)
    expected_x, expected_y = (-0.9999, -1.9999)
    assert math.isclose(result_x, expected_x, abs_tol=1e-3)
    assert math.isclose(result_y, expected_y, abs_tol=1e-3)
