'''mllib.py:
Python module implementing some machine learning algorithms learned in
Stanford's Machine Learning course on Coursera (October 2015):
https://www.coursera.org/learn/machine-learning/
'''

import matlib
import math
import sys
from datetime import datetime

# System epsilon value
epsilon = sys.float_info.epsilon



class MLError(Exception):
    '''Exception alias MLError (for [M]achine [L]earning Error).'''
    pass

# Some common instances (can be internalised to specific functions).
InvalidData = MLError("Invalid data provided")
InvalidAlpha = MLError("Learning rate must be a positive value")
InvalidLambda = MLError("Regularisation rate must be a positive value or nil")
CalcFail = MLError('Calculation failed!')

class DivergentAlpha(Exception):
    '''Special alias for Exception raised when an algorithm using a value alpha
    is divergent.
    '''
    pass

'''Vectorised linear regression hypothesis function.'''
linh = lambda T, X: X * T
'''Vectorised linear regression cost (error) function.'''
linj = lambda h, T, X, y: (h(T, X) - y).ssq / (2*X.Rows)
'''Sigmoid function'''
sigmoid = lambda Z: Z.apply(lambda z: 1/(1 + matlib.exp(-z)))
'''Vectorised logistic regression hypothesis function.'''
logh = lambda T, X: sigmoid(X*T)
'''Vectorised logistic regression cost (error) function. Not implemented.'''
logj = lambda h, T, X, y: \
       (-y.T*matlib.log(h(T, X)) - (1-y).T*matlib.log(1-h(T, X))) / X.Rows

def gd(data, alpha, reglambda=0, nl=1, initvalues=None, maxiter=1E5,
       hypothesis=linh, cost=linj):
    '''Generalised gradient descent algorithm.
    
    Based on contour plots, gradient descent tries to find the parameters
    that minimise the cost of a hypothesis function to model given data by
    iteratively calculating gradients and converging towards a minimum.
    Default values assume a linear regression.
    
    Args:
        data (matlib.Matrix): Data matrix, must contain at least 2 columns.
            The last column is treated as the training output values vector.
        alpha (float): Learning rate alpha must be a positive nonzero value.
        reglambda (float): Regularisation rate lambda, must be a positive
            or nil value. Defaults to 0.
        initvalues (matlib.Vector): Initial parameters vector. Defaults to
            None which initialises to a Zeroes vector with the same number of
            rows as the ``data`` matrix.
        maxiter (int): Maximum number of iterations for the algorithm.
            Defaults to 100,000 iterations. It can be assigned None, which
            keeps iterating indefinetly, but it's not recommended.
        hypothesis (callable): Parametrised hypothesis function. Must accept
            2 arguments. Defaults to the linear regression hypothesis
            function.
        cost (callable): Parametrised cost function. Must accept 4 arguments.
            Defaults to the linear regression cost function.

    Raises:
        InvaliData if no data is provided or data has less than 2 columns.
        InvalidAlpha if learning rate is less or equal 0.
        InvalidLambda if regularisation rate is negative.
        DivergentAlpha if the algorithm diverges.

    Returns:
        (matlib.Vector, float): The parameters that minimise the cost of the
            hypothesis function and the value of the cost for those
            parameters.
    '''
    # ``data`` must be a Matrix with a minimum of 2 columns.
    if not isinstance(data, matlib.Matrix) or not data or data.Columns < 2:
        raise InvalidData
    # Number of training examples m == ``data`` matrix rows
    # Number of parameters to find n == ``data`` matrix columns (technically,
    #>it's the number of columns - 1 for the result vector and + 1 for the bias,
    #>hence the number of parameters is the number of columns)
    m, n = data.Dimension
    # ``alpha`` must be a positive real number
    alpha = float(alpha)
    if alpha <= 0.0:
        raise InvalidAlpha
    # Regularisation rate must be a positive real number or nil
    l = float(reglambda)
    if l < 0.0:
        raise InvalidLambda
    # Design and result matrices
    X = matlib.Ones(data.Rows, 1) | matlib.Matrix(*data[:,:-1])
    y = matlib.Vector(*data[:,-1])
    # Initial values
    ZV = matlib.Vector(matlib.Zeroes(n, 1)) # Zero vector
    if not initvalues:
        T = ZV
    else:
        # Vectorise if not a vector
        T = matlib.Vector(initvalues) \
            if not isinstance(initvalues, matlib.Vector) else initvalues
        # Use the Zero vector if wrong dimensions are provided
        T = ZV if not T.Dimension == (n, 1) else T
    # Initialise
    N, rss, converged, maxiter = 0, None, False, abs(int(maxiter))
    divergent = DivergentAlpha("Gradient descent diverges for alpha=%s" % alpha)
    l1, l2 = matlib.Ones(*T.Dimension), matlib.Identity(T.Rows)
    l1[0, 0], l2[0, 0] = 0, 0
    j1, j2 = None, None
    # Iterate
    while (not maxiter or N < maxiter) and not converged:
        # Increment iterations
        N += 1
        # Simultaneous update, keep old values in ``d`` with regularisation
        d, T = T.apply(), T - alpha/m * (X.T*(hypothesis(T, X) - y) + l*l2*T)
        # Check for convergence:
        j1 = j2 if N > 1 else None
        # If no cost function is provided, use the empirical formula
        #>cost = (d - T).ssq / 2
        j2 = (cost(hypothesis, T, X, y) if cost else (d - T).ssq/2) \
             + l * l1.T * T.apply(lambda x: x**2) / (2*m) # Regularisation
        # The cost should decrease for convergence (thus the algorithm diverges
        #>when ``jdiff`` is positive or 0).
        jdiff = -2*epsilon if j1 is None else j2 - j1
        if jdiff >= 0:
            raise divergent
        # The algorithm is considered converged when ``jdiff`` is too small.
        converged = abs(jdiff) <= epsilon
    # Return
    return matlib.Vector(T), j2

def normeq(data, reglambda=0):
    '''Normal equation:

    Method to calculate values of unknowns with regularisation by applying:
        (X^T*X + lambda*[I(n+1) with I(0,0)=0])^-1 * X^T * y


    Args:
         data (matlib.Matrix): Data matrix, must contain at least 2 columns.
             The last column is treated as the training output values vector.
        reglambda

    Raises:
        InvaliData if no data is provided or data has less than 2 columns.

    Returns:
        matlib.Vector: The resulting parameters vector.
    '''
    # ``data`` must be a Matrix with a minimum of 2 columns.
    if not isinstance(data, matlib.Matrix) or not data or data.Columns < 2:
        raise InvalidData
    # Design and result matrices
    X = matlib.Ones(data.Rows, 1) | matlib.Matrix(*data[:,:-1])
    y = matlib.Vector(*data[:,-1])
    # Regularistion rate must be positive or 0
    l = float(reglambda)
    if l < 0.0:
        raise InvalidLambda
    I = matlib.Identity(X.Columns)
    I[0, 0] = 0
    # Calculate the params
    T = ~(X.ssq + l*I) * X.T * y
    # Return a Vector object
    return  matlib.Vector(T)

def linreg(data, algorithm=None, alpha=None, reglambda=0, maxiterations=1E5,
           zero=1E-6, returntype=matlib.Vector):
    '''Generic multivariate linear regression method.

    Determine the parameters that regress the given ``data`` linearly.

    Args:
        data (matlib.Matrix): Data matrix, must contain at least 2 columns.
            The last column is treated as the training output values vector.
        algorithm (str): Algorithm to use for regression. Possible values are:
            % "ne": Normal equation algorithm. This is the default algorithm.
            % "gd": gradient descent algorithm.
        alpha (float): Learning rate alpha must be a positive nonzero value.
            Applicable only for gradient descent? Defaults to 1. This is the
            starting value for alpha, if it diverges, it's reduced by sqrt(10)
            until the algorithm converges.
        reglambda (float): Regularisation rate lambda, must be a positive
            or nil value. Defaults to 0.
        maxiterations (int): Maximum number of iterations for the algorithm.
            Defaults to 100,000 iterations. It can be assigned None, which
            keeps iterating indefinetly, but it's not recommended.
        zero (float): Precision value. Value below which everything is rounded 
            to 0. Also used to determine the maximum number of decimals.
            Defaults to 1E-6 (resulting in 6 decimal digits).
        returntype (callable): Transforms the parameters. Defaults to
            matlib.Vector, transforming the paramters into a Vector.

    Raises:
        mllib.MLError: raised when calculations fail for various alpha values
            for gradient descent (limited by provided ``zero`` value).

    Returns:
        (``returntype``, float): The transformed parameters list and the cost
            of the regression.
    '''
    # "Matricise" ``data``
    data = matlib.Matrix(data) if not isinstance(data, matlib.Matrix) else data
    # Determine algorithm, defaults to "ne"
    algorithm = "ne" if algorithm is None or algorithm not in ("gd", "ne") \
                else algorithm
    # Precision
    zero = float(zero) if zero else 1E-6  # We must have a value =/= exactly 0
    pr = int(round(-math.log10(zero)))
    # Regularisation rate defaults to 0
    l = float(reglambda) if reglambda else 0.0
    # Initialise:
    T, J = None, None
    # Use specified algorithm in calculation
    if algorithm == "ne":
        T = normeq(data, reglambda=l)
        # Calculate cost
        X = matlib.Ones(data.Rows, 1) | matlib.Matrix(*data[:,:-1])
        y = matlib.Vector(*data[:,-1])
        J = linj(linh, T, X, y)
    elif algorithm == "gd":
        '''Technically it can be just an "else" since I limited the values to
        ("gd", "ne"), but kept this way just in case more algorithms were added.
        '''
        # Default ``alpha`` is 1, divide by sqrt(10) each subsequent iteration
        alpha, alphastep = float(alpha) if alpha else 1, math.sqrt(10)
        # Attempt loop
        converged = False
        while not converged:
            if alpha <= zero: # 2*pr attempts max
                raise CalcFail
            try:
                T, J = gd(data, alpha=alpha, reglambda=l, maxiter=maxiterations)
                converged = True
            # Handle divergent alpha
            except DivergentAlpha:
                alpha /= alphastep
            # Forward other errors
            except:
                raise
    # Zeroing
    round_value = lambda t: 0.0 if abs(t) < zero else round(t, pr)
    T, J = T.apply(round_value), round_value(J)
    # Return
    return returntype(T), J
    
def logreg(data, algorithm=None, alpha=None, reglambda=0, maxiterations=1E5,
            zero=1E-6, returntype=matlib.Vector):
    '''Binary logistic regression method using gradient descent (for now).

    Args:
        data (matlib.Matrix): Data matrix, must contain at least 2 columns.
            The last column is treated as the training output values vector.
            alpha (float): Learning rate al starting value for alpha, if it
            diverges, it's reduced by sqrt(10) until the algorithm converges.
        algorithm (str): Algorithm to use for regression. Possible values are:
            % "op": Using optimisation function. This is the default algorithm.
                 Not yet implemented until fmincg is implemented.
            % "gd": gradient descent algorithm.
        alpha (float): Learning rate alpha must be a positive nonzero value.
            Applicable only for gradient descent? Defaults to 1. This is the
            starting value for alpha, if it diverges, it's reduced by sqrt(10)
            until the algorithm converges.
        reglambda (float): Regularisation rate lambda, must be a positive
            or nil value. Defaults to 0.
        maxiterations (int): Maximum number of iterations for the algorithm.
            Defaults to 100,000 iterations. It can be assigned None, which
            keeps iterating indefinetly, but it's not recommended.
        zero (float): Precision value. Value below which everything is rounded 
            to 0. Also used to determine the maximum number of decimals.
            Defaults to 1E-6 (resulting in 6 decimal digits).
        returntype (callable): Transforms the parameters. Defaults to
            matlib.Vector, transforming the paramters into a Vector.

    Raises:
        mllib.MLError: raised when calculations fail for various alpha values
            for gradient descent (limited by provided ``zero`` value).

    Returns:
        (``returntype``, float): The transformed parameters list and the cost
            of the regression.
    '''
    # "Matricise" ``data``
    data = matlib.Matrix(data) if not isinstance(data, matlib.Matrix) else data
    # Validate y in { 0, 1 }
    for y in data[:,-1]:
        if y not in (0, 1):
            NotBinary = MLError("Classes data must be binary!")
            raise NotBinary
    # Determine algorithm, defaults to "op"
    algorithm = "op" if algorithm is None or algorithm not in ("gd", "op") \
                else algorithm
    # Precision
    zero = float(zero) if zero else 1E-6  # We must have a value =/= exactly 0
    pr = int(round(-math.log10(zero)))
    # Regularisation rate defaults to 0
    l = float(reglambda) if reglambda else 0.0
    # Initialise:
    T, J = None, None
    # Use specified algorithm in calculation
    if algorithm == "op":
        return NotImplemented
    elif algorithm == "gd":
        '''Technically it can be just an "else" since I limited the values to
        ("gd", "op"), but kept this way just in case more algorithms were added.
        '''
        # Default ``alpha`` is 1, divide by sqrt(10) each subsequent iteration
        alpha, alphastep = float(alpha) if alpha else 1, math.sqrt(10)
        # Attempt loop
        converged = False
        while not converged:
            if alpha <= zero: # 2*pr attempts max
                raise CalcFail
            try:
                T, J = gd(data, alpha=alpha, reglambda=l, maxiter=maxiterations,
                          hypothesis=logh, cost=logj)
                converged = True
            # Handle divergent alpha
            except DivergentAlpha:
                alpha /= alphastep #if not converged else 1
            # Forward other errors
            except:
                raise
    # Zeroing
    round_value = lambda t: 0.0 if abs(t) < zero else round(t, pr)
    T, J = T.apply(round_value), round_value(J)
    # Return
    return returntype(T), J
