'''matlib.py:
Python module implementing basic matrix operations needed for Stanford's 
Machine Learning course on Coursera (October 2015):
https://www.coursera.org/learn/machine-learning/
'''

import math
import random
import sys

'''
TODO: Types matching is not perfect for all operations. Some work should be done
to improve it.
'''

class MatrixError(Exception):
    '''MatrixError subclass of Exception'''
    pass

class Matrix(object):
    '''Base Matrix type.

    Constructor:
        Args:
            *data: Each row as a list (or a tuple) or a Matrix.
                 The length of the first list is considered the number of
                 columns of the matrix and the subsequent rows (lists/tuples)
                 are validated against it. A Matrix can be embedded if it
                 has the same number columns as the detected number of columns.
        Examples:
            A = matlib.Matrix([1, 2, 3]) # Row as a list
            B = matlib.Matrix((1, 2, 3), (4, 5, 6)) # Rows as tuples
            C = matlib.Matrix([6, 7, 8], A) # Matrix embedding

    Attributes:
        __matrix__ (list): Linear representation of the matrix as list of lists

    0-indexed item access and assignment:
        % MatrixInstance[idx]       <-> Row $idx as a list
        % MatrixInstance[r1:r2]     <-> List of rows $r1 to $r2 as lists
        % MatrixInstance[r,c]       <-> Item at row $r and column $c as a scalar
        % MatrixInstance[:, idx]    <-> Column $idx as a list
        % MatrixInstance[:, c1:c2 ] <-> List of columns $c1 to $c2 as lists

    Matlab-style linear 0-indexed item access:
       % MatrixInstance(idx)
             -> MatrixInstance[idx mod self.Columns, idx div self.Columns]
    '''
    def __init__(self, *data):
        '''Constructor.'''
        # Validate data
        if not data:
            raise MatrixError("No data provided!")
        # Initialise
        self.__matrix__, ncol = [], None
        # Process data
        for d in data:
            # Type validation
            DataTypeError = TypeError("%s has an invalid data type." % str(d))
            if type(d) not in (tuple, list) and not isinstance(d, Matrix):
                raise DataTypeError
            # List of all rows if a Matrix or just the current row if a simple
            #>list/tuple
            allrows = d.__matrix__ if isinstance(d, Matrix) else [ d ]
            # Insert all rows (one row if simple mode)
            for row in allrows:
                # Force list type to store in __matrix__
                row = list(row)
                # Length validation to first list's length
                ncol = len(row) if not ncol else ncol
                DataSizeError = MatrixError("Invalid row %s size. %s detected."
                                            % (row, ncol))
                if len(row) != ncol:
                    raise DataSizeError
                '''Check that all row items are numbers by dividing it by 1.0,
                this converts all numbers to float type, and would raise an
                an error if it's not a number. Of course this isn't duck
                typing proof. A possible workaround for this issue could be to
                additionally test +, -, * and ** but for now the division
                is sufficient.
                '''
                #all_numbers = [ (n/1.0, n+0, n-0, n*1, n**1) for n in row ]
                all_numbers = [ n/1.0 for n in row ]
                # Append to list of lists
                #self.__matrix__.append([ t[0] for t in all_numbers ])
                self.__matrix__.append(all_numbers)

    def __iter__(self):
        '''Iterator over matrix rows as lists.'''
        return iter(self.__matrix__)

    @staticmethod
    def Key(key, w, h):
        '''Static method to validate and parse a given matrix key.
        
        A valid key is either an integer, a slice or a 2-tuple combining both.

        Args:
            key (int or tuple or slice): The key to parse.
            w (int): Matrix number of rows to consider in relative calculations.
            h (int): Matrix number of columns to consider in relative
                calculations.

        Returns:
            (int or slice, int or slice): The validated key, a combination of
                int and slice indices.

        Raises:
            TypeError: Custom type error when the format of the key or any of
                the indices is invalid.
        '''
        keytype, InvalidKey = type(key), TypeError("Invalid key %s" % repr(key))
        # Validate type
        if keytype not in (int, tuple, slice):
            raise InvalidKey
        # Parse into row and column keys
        r, c = key if keytype is tuple else (key, slice(None))
        rtype, ctype = type(r), type(c)
        # Row and column keys must be either int or slice at this stage
        if rtype not in (int, slice) or ctype not in (int, slice):
            raise InvalidKey
        # Work around negative indices by transforming them relatively to the
        #>provided width and height. Slices are kept since they test positively
        #>to being greater or equal to 0.
        r, c = r if r>=0 else w + r, c if c>=0 else h + c
        # Return row and columns keys as a 2-tuple
        return r, c

    def __getitem__(self, key):
        '''Item getter.'''
        # Parse the key using the static Matrix.Key method
        r, c = Matrix.Key(key, *self.Dimension)
        #The row index type determines the behaviour: if it's an integer, we
        #>return the corresponding row list limited by the column index; if it's
        #>a slice we return, we returning a list limited by the row slice of
        #>row lists limited by the column index.
        return (self.__matrix__[r][c] if type(r) == int
                else [ row[c] for row in self.__matrix__[r] ])

    def __setitem__(self, key, value):
        '''Item setter.'''
        # Parse the key using the static Matrix.Key method
        r, c = Matrix.Key(key, *self.Dimension)
        rtype, ctype = type(r), type(c)
        # Direct item assignment must be straightforward, although it converges
        #>with the other method. Assigned value must be a scalar (division by
        #>1.0, check constructor for details on scalar validation).
        if rtype == ctype == int:
            return self.__matrix__[r].__setitem__(c, value/1.0)
        # If not a direct item assignment, value must be list or tuple
        InvalidValue = Exception("Cannot assign value %s %s" % (repr(value),
                                                                type(value)))
        # Do I need to test the indices? Needs confirmation since, at this
        #>stage, rows index being an integer and columns index being an integer
        #>are mutually exclusive.
        #if not rtype == ctype == int and type(value) not in (list, tuple):
        if type(value) not in (list, tuple):
            raise InvalidValue
        # Change both indices to slices (negative indices are caculated
        #>relatively to dimensions).
        w, h = self.Rows, self.Columns
        xr = r if rtype is slice else slice(r if r >= 0 else w + r, 1)
        xc = c if ctype is slice else slice(c if c >= 0 else h + c, 1)
        # Determine assignment ranges limits based on slices
        rmin = min(xr.start if xr.start > 0 else 0, w - 1) # First row
        rmax = min(rmin + (xr.stop or w), w) # Last row
        cmin = min(xc.start if xc.start > 0 else 0, h - 1) # First column
        cmax = min(cmin + (xc.stop or h), h) # Last column
        # Copy self.__matrix__ to avoid writing in case of errors
        matrix = [ [ c for c in r ] for r in self.__matrix__ ]
        # Map value to found ranges
        for i in xrange(rmin, rmax):
            # In single row assignment, take the provided value as is. It is
            #>required to be a list of scalars. In multiple rows assignment, we
            #>map relatively to the range limits.
            newrow = value if rtype == int else value[ i - rmin ]
            for j in xrange(cmin, cmax):
                # In single column assignment, take the row value as is.
                #>It is required to be list of a scalars. In multiple columns
                #>assignment, we map relatively to the range limits.
                val = newrow if ctype == int else newrow[j - cmin]
                # Value must be scalar
                matrix[i][j] = val / 1.0
        # Store new value on success
        self.__matrix__ = matrix

    def __call__(self, idx):
        '''Matlab-style linear 0-index item access

        When an instance is called with an integer as a parameter, it will be
        considered as the linear 0-index value calculated over rows.

        Usage:
            A = matlib.Matrix([1, 2, 3, 4], [5, 6, 7, 8])
            A(0) == A[0,0] == 1
            A(1) == A[1,0] == 5
            A(2) == A[0,1] == 2
            A(7) == A[1,3] == 8

        Args:
            idx (int): linear 0-index

        Returns:
            float: Value of item at index
        '''
        # Index must be an integer
        orgidx = int(idx)
        # Maximum number of items
        nitems = self.Rows * self.Columns
        # Adjust relative index and validate
        idx = int(orgidx) if orgidx >= 0 else nitems + int(orgidx)
        if idx < 0 or idx >= nitems:
            InvalidIndex = MatrixError("Index %s out of range!" % orgidx)
            raise InvalidIndex
        # Number of rows used to calculate index in list:
        #>Row = idx mod rows, Column = idx div rows
        r = self.Rows
        return self.__matrix__[idx % r][idx // r]

    def __nonzero__(self):
        '''Evaluate non zero.

        Usage:
            B = not A

        Returns:
            bool: False if self.__matrix__ is empty, True otherwise.
        '''
        return bool(self.__matrix__)

    def __neg__(self):
        '''Negation:

        -A = A * -1

        Usage:
            B = -A

        Returns:
            matlib.Matrix: Resulting negated matrix.
        '''
        return self.__mul__(-1)

    def __add__(self, other):
        '''Addition:

        Matrix addition consists of the following cases:
            % Matrix + Matrix: Matrices should havesame dimensions for
                  element-wise addition
            % Matrix + Scalar: The scalar is added to each element of the Matrix
        Addition is commutable (see __radd__).

        Usage:
            B = A + other
        
        Args:
            other (matlib.Matrix or int or float): the right-hand operand of the
                addition operation.

        Returns:
            matlib.Matrix: Matrix resulting from the addition.
        '''
        # If the current instance is a scalar, call other's __add__ method by
        #>commuting the operands.
        if self.isScalar:
            return other + self.__matrix__[0][0]
        # First case: Matrix + Matrix
        scalar = None
        if isinstance(other, Matrix):
            # We face two cases here: the other matrix is a multi-dimensional
            #>matrix or a scalar.
            if not other.isScalar and self.Dimension == other.Dimension:
                # Multi-dimensional matrices element-wise addition
                newdata = [ [ s+o for s,o in zip(rs, ro) ]
                            for rs,ro in zip(self, other) ]
                # Return a Matrix object
                return Matrix(*newdata)
            elif other.isScalar:
                # We use the scalar value
                scalar = other.__matrix__[0][0]
            else:
                DimensionMismatch = MatrixError(
                    "Cannot add matrices of different dimensions %s + %s"
                    % (self.Dimension, other.Dimension))
                raise DimensionMismatch
        # Second case: Matrix + Scalar
        scalar = scalar or other/1.0
        # Add the scalar value to all elements
        return self.apply(lambda i: i + scalar)
    
    def __radd__(self, other):
        '''Reverse addition:

        Since addition is commutable (A + B = B + A) then this is only an alias
        for the __add__ method to be able to handle expression such as
        Number + Matrix (or any other type that implements addition).

        Usage:
            B = other + A

        For details, check __add__ docstring.
        '''
        return self.__add__(other)
    
    def __sub__(self, other):
        '''Substraction:

        A - B = A + (-B) thus substraction amounts to the summation with the
        negated other value.

        Usage:
            B = A - other

        For details, check __add__ and __neg__ docstrings.
        '''
        return self.__add__(-other)

    def __rsub__(self, other):
        '''Reverse substraction:
        
        Since substraction amounts to the summation with the negated value,
        we negate the current instance and add it to ``other``. This would be
        equivalent to calling ``other`` __add__ implementation or
        Matrix.__radd__ implementation in the negated Matrix.

        Usage:
            B = other - A

        For details, check __radd__ and __neg__ docstrings.
        '''
        return other + self.__neg__()
    
    def __mul__(self, other):
        '''Multiplication:

        Matrix multiplication consists of the following cases:
            % Matrix * Matrix: The multiplication is the sum of the
                 element-wise product of each row in the first matrix with each
                 column of the second matrix, thus the number of columns of the
                 first matrix should match the number of rows of the second
                 matrix. The resulting has the same number of rows as the first
                 matrix and the same number of columns as the second matrix.
            % Matrix * Scalar: The scalar is multiplied by each element of the
                matrix.
        Multiplication is commutable for scalar multiplication but not for
        matrices multiplication (see __rmul__).

        Usage:
            B = A * other
        
        Args:
            other (matlib.Matrix or int or float): the right-hand operand of the
                multiplication operation.

        Returns:
            matlib.Matrix: Matrix resulting from the multiplication.
        '''
        # If the current instance is a scalar, call other's __mul__ method by
        #>commuting the operands.
        if self.isScalar:
            return other * self.__matrix__[0][0]
        # First case: Matrix * Matrix
        scalar = None
        if isinstance(other, Matrix):
            # We face two cases here: the other matrix is a multi-dimensional
            #>matrix or a scalar.
            if not other.isScalar and self.Columns == other.Rows:
                # Multi-dimensional matrices multiplication
                # Transpose ``other`` so we can iterate over the columns
                ot = other.transpose()
                # Sums of products
                newdata =  [ [ sum(a*b for a,b in zip(rs, co)) for co in ot ]
                             for rs in self ]
                # Return a Matrix object
                return Matrix(*newdata)
            elif other.isScalar:
                # We use the scalar value
                scalar = other.__matrix__[0][0]
            else:
                DimensionMismatch = MatrixError(
                "Cannot multiply matrices with mismatched dimensions %s * %s"
                % (self.Dimension, other.Dimension))
                raise DimensionMismatch
        # Second case: Matrix + Scalar
        scalar = scalar or other/1.0
        # Multiply the scalar value to all elements
        return self.apply(lambda i: i * scalar)

    def __rmul__(self, other):
        '''Reverse multiplication:

        Since ultiplication is commutable for scalar multiplication but not for
        matrices multiplication (A * B =/= B * A; A * k = k * A)
        then this is only an alias for the __mul__ method to be able to handle
        expressions such as Number * Matrix (or any other type that implements
        multiplication). Returns NotImplemented in the case of Matrix * Matrix
        to let the __mul__ method take over.

        Usage:
            B = other * A

        For details, check __mul__ docstring.
        '''
        return NotImplemented \
            if isinstance(other, Matrix) else self.__mul__(other)

    def __div__(self, other):
        '''Division:

        Generally speaking, division is the multiplication by the inverse
        (A / B = A * B^-1). In the case of matrices it's the inverse matrix and
        in the case of scalars it's 1/scalar.

        Thus division is an alias for __mul__ where the second operand is taken
        to the power -1 (check __pow__ for Matrix instances)

        Usage:
            B = A / other

        For details, check __mul__ docstring
        '''
        return self.__mul__(pow(other, -1))

    def __rdiv__(self, other):
        '''Reverse division:
        
        Usage:
            B = other / A
        
        Same as forward division (check __div__ docstring). The inverted however
        is the current instance.
        '''
        return other * pow(self, -1)

    def __pow__(self, other):
        '''Exponentiation:

        A**p: p < 0, invert and raise to abs(p)
              p > 0: multiply square matrix p times itself (p is integer)
              p = T: transpose

        Usage:
            B = A ** other
            B = pow(A, other)

        Returns:
            matlib.Matrix: the resulting matrix
            or
            float: if the Matrix is scalar
        '''
        p = other if other != 'T' else 1
        # Act on transposed version if 'T' else current instance
        G = self if other != 'T' else self.transpose()
        # Negative power: Invert and raise to the absolute value
        if p < 0:
            return pow(G.__invert__(), abs(p)) # ~G ** abs(p)
        # For scalar values, use the numerical pow method
        if G.isScalar:
            return pow(G.__matrix__[0][0], p)
        # For multi-dimensional square matrices, p can only be integers
        #>N.B. There is an eigenvalues method, for non-integer powers, but I
        #>have not implemented it.
        if p != int(p):
            return NotImplemented
        if p > 1 and G.Rows != G.Columns:
            NonSquareMatrix = MatrixError('Matrix must be square!')
            raise NonSquareMatrix
        # Using the general formula A^(n+1) = A^n*A
        return reduce(lambda x, i: x * G, xrange(p), 1)

    def __or__(self, other):
        '''Horizontal matrix concatenation using the bitwise or operator:

        Concatenates two matrices with the same number of rows resulting in a
        matrix with the same number of rows and the sum of the two number of
        columns. If a scalar value is provided, a column with all elements as
        the scalar value is appended to the Matrix.

        Usage:
            C = A | B

        Args:
            other (matlib.Matrix or int): Matrix or scalar to concatenate to
               current Matrix instance.

        Returns:
            matlib.Matrix: resulting concatenated Matrix
        '''
        # Transform scalar values into a column of scalars with the same number
        #>of rows as the current Matrix. Scalar validation, as above, is done
        #>using the division by 1.0 method (which is good enough for now).
        other = other if isinstance(other, Matrix) \
                else Matrix(*[ [ other/1.0 ] ]*self.Rows)
        if self.Rows != other.Rows:
            SizeMismatch = MatrixError(
                "Cannot concatenate matrices with different rows %s | %s"
                % (self.Dimension, other.Dimension))
            raise SizeMismatch
        return Matrix(*[ r1 + r2 for r1, r2 in zip(self, other)  ])

    def __and__(self, other):
        '''vertical matrix concatenation using the bitwise and operator:

        Concatenates two matrices with the same number of columns resulting in a
        matrix with the same number of columns and the sum of the two number of
        rows. If a scalar value is provided, a row with all elements as
        the scalar value is appended to the Matrix.

        Usage:
            C = A & B

        Args:
            other (matlib.Matrix or int): Matrix or scalar to concatenate to
               current Matrix instance.

        Returns:
            matlib.Matrix: resulting concatenated Matrix
        '''
        # Transform scalar values into a row of scalars with the same number
        #>of columns as the current Matrix. Scalar validation, as above, is done
        #>using the division by 1.0 method (which is good enough for now).
        other = other if isinstance(other, Matrix) \
                else Matrix([ other/1.0 ] * self.Columns)
        if self.Columns != other.Columns:
            SizeMismatch = MatrixError(
                "Cannot concatenate matrices with different columns %s | %s"
                % (self.Dimension, other.Dimension))
            raise SizeMismatch
        # I exploit the Matrix constructor for the concatenation
        return Matrix(self, other)
    
    def transpose(self):
        '''Transpose Matrix:

        Transposition transforms the Matrix rows into columns and columns into
        rows. It follows the general formula A^T[i, j] = A[j, i].

        Usage:
            B = A.transpose()
           # Other aliases via the __pow__ function:
           B = A ** 'T'
           B = pow(A, 'T')
           # Other alias via the __reversed__ function:
           B = reversed(A)
           # Other alias using the T property:
           B = A.T

        Returns:
            matlib.Matrix: the transposed Matrix
        '''
        # Version with reduced key calculation overhead, working directly on
        #>the __matrix__ list which should have a better performance.
        return Matrix(*[ [ row[i] for row in self.__matrix__ ]
                         for i in xrange(self.Columns) ])
        # Obsoleted working version, depending on the item getter syntax.
        return Matrix(*[ self[:,i] for i in xrange(self.Columns) ])
    
    def __reversed__(self):
        '''Alias for transpose using the reversed(A) built-in function call.'''
        return self.transpose()
    
    def __invert__(self, pseudo=True):
        '''Matrix inversion:

        The inverse matrix is the matrix that results in the identity matrix
        when multiplied by the original matrix (A^-1 * A = I). We can
        distinguish 3 cases:
            % Scalar mode: The inverse of a scalar is a scalar equal to the
                scalar value raised to the power -1.
            % Square-matrix mode: Invert using the Gauss-Jordan algorithm.
                Current implementation was inspired by code at [http://ricardianambivalence.com/2012/10/20/pure-python-gauss-jordan-solve-ax-b-invert-a/]
            % Pseudo-inverse mode: Invert using the generic Moore-Penrose
                pseudoinverse algorithm. Current implementation was based on
                the paper at [http://gsite.univ-provence.fr/gsite/Local/lpc/dir/courrieu/articles/Courrieu05b.pdf]

        Usage:
           B = A.__invert__(pseudo)
           B = ~A
           # Other aliases via the __pow__ function:
           B = A ** -1
           B = pow(A, -1)

        Args:
           pseudo (bool): Flag to indicate whether to use the pseudo-inverse
              method. True by default.

        Returns:
           matlib.Matrix: Inverted Matrix
           or
           float: Inverted scalar
        '''
        # Scalar mode
        if self.isScalar:
            return Matrix([pow(self.__matrix__[0][0], -1)])
        # Multi-dimensional Matrix mode
        if pseudo == False and self.Rows == self.Columns:
            '''Invert using the Gauss-Jordan algorithm if square matrix.'''
            # Append Identity matrix and transpose
            m = (self & Identity(self.Rows)).T
            #
            eqns, colrange, augCol = self.Rows, self.Rows, m.Columns
            # Permute the matrix -- get the largest leaders onto the diagonals.
            #>Take the first row, assume that x[1,1] is largest, and swap if
            #>that's not true.
            for col in xrange(colrange):
                bigrow = col
                for row in xrange(col+1, colrange):
                    if abs(m[row][col]) > abs(m[bigrow][col]):
                        bigrow = row
                        m[col], m[bigrow] = m[bigrow], m[col]
            # Reduce, such that the last row has at most one unknown.
            for rrcol in xrange(0, colrange):
                for rr in xrange(rrcol+1, eqns):
                    cc = -(float(m[rr,rrcol])/float(m[rrcol,rrcol]))
                    for j in xrange(augCol):
                        m[rr,j] = m[rr,j] + cc*m[rrcol,j]
            # Final reduction -- the first test catches under-determined
            #>systems. These are characterised by some equations being all zero.
            for rb in reversed(xrange(eqns)):
                if ( m[rb,rb] == 0):
                    if m[rb,augCol-1] == 0:
                        continue
                    elif pseudo:
                        return self.__invert__()
                    else:
                        raise MatrixError('System is inconsistent')
                else:
                    # you must loop back across to catch under-determined
                    #>systems
                    for backCol in reversed(xrange(rb, augCol)):
                        m[rb,backCol] = float(m[rb,backCol])/float(m[rb,rb])
                    # knock-up (cancel the above to eliminate the knowns)
                    # again, we must loop to catch under-determined systems
                    if not (rb == 0):
                        for kup in reversed(xrange(rb)):
                            for kleft in reversed(xrange(rb, augCol)):
                                kk = -float(m[kup,rb]) / float(m[rb,rb])
                                m[kup,kleft] += kk*float(m[rb,kleft])
            # Return
            return Matrix(*m[:,augCol/2:]).T
        '''Invert using the generic Moore-Penrose pseudoinverse algorithm.'''
        G = self
        # Adjust parameters
        trans, n = G.Rows < G.Columns, min(*G.Dimension)
        # A = M^T' * M or M * M^T, (n x n)
        A = G * G.T if trans else G.ssq
        # Full rank Cholesky factorization of A
        r, tol = 0, min(filter(lambda l: l > 0, A.Diagonal)) * 1e-9
        L= Zeroes(*A.Dimension) # Float [0.0] (n x n)
        for k in xrange(n):
            r += 1
            V = Matrix(A[k:n, k]) - (Matrix(*L[k:n,:r]) * Matrix(L[k,:r]).T).T
            L[k:n, r-1] = [V] if type(V) in (float, int) else V[0]
            di = L[k, r-1]
            if di > tol:
                L[k, r-1] = math.sqrt(di)
                if k < n-1:
                    T = Matrix(L[k+1:n,r-1]) / L[k,r-1]
                    L[k+1:n,r-1] = [T] if type(T) in (float, int) else T[0]
            else:
                r -= 1
        # Computation of the generalised inverse
        L = Matrix(*L[:,:r])
        M = L.ssq.__invert__(pseudo=False)
        Y  = G.T * L * M * M * L.T if trans else L * M * M * L.T * G.T
        # Return
        return Y
    
    def apply(self, fn=lambda l: l, *args, **kw):
        '''Apply a function to every element of the Matrix.

        The default call of this function can be used as a method to deep-copy
        the matrix. If the function doesn't return a numerical value, the
        function would fail when constructing the new Matrix object.

        Usage:
           B = A.apply() # Copies the Matrix
           B = A.apply(fn, *args, *kw)

        Args:
            fn (callable): Function to be applied to Matrix elements. By default
                it returns the element itself.
            *args, **kw: lists of arguments to pass to the callable function
                in addition to the element.

        Returns:
            matlib.Matrix: the transformed Matrix.
        '''
        return Matrix(*[ [ fn(el, *args, **kw) for el in r ] for r in self ])

    @property
    def Rows(self):
        '''Number of rows.'''
        return len(self.__matrix__)
    
    @property
    def Columns(self):
        '''Number of columns.'''
        return len(self.__matrix__[0])
    
    @property
    def Dimension(self):
        '''(Number of rows, Number of columns) 2-tuple.'''
        return (self.Rows, self.Columns)
    
    @property
    def isScalar(self):
        '''Scalar status. True if matrix dimension is (1, 1).'''
        return (self.Dimension == (1, 1))
    
    @property
    def T(self):
        '''Transposed matrix.'''
        return self.transpose()
    
    @property
    def Diagonal(self):
        '''List of elements on the diagonal.'''
        return [ r[i] for r, i in zip(self, xrange(min(*self.Dimension))) ]
    
    @property
    def Inverse(self):
        '''Inverse matrix.'''
        return self.__invert__()
    
    @property
    def MaximumValue(self):
        '''Maximum value in the matrix.'''
        return max(max(self.__matrix__))
    
    @property
    def MinimumValue(self):
        '''Minimum value in the matrix.'''
        return min(min(self.__matrix__))
    
    @property
    def ssq(self):
        '''Symetrical square matrix:

        Useful matrix made by the product of the transposed matrix by the matrix
        itself (A^T * A), resulting in a square (number of columns x number of
        columns) and symetrical matrix.
        '''
        return self.transpose() * self
    
    def __str__(self):
        '''String representation:

        Scalar are represented as numbers.
        For matrices, we find the longest numerical representation length for
        all values and apply it to all numbers to display equal-length columns
        over all rows.
        '''
        if self.isScalar:
            return str(self.__matrix__[0][0])
        maxlength = max([max([len(str(l)) for l in r]) for r in self]) + 1
        return "\n".join([" ".join(["%{0}s".format(maxlength) % l
                                       for l in r]) for r in self])
    def __repr__(self):
        return self.__str__()

class Vector(Matrix):
    '''Vector subclass of Matrix:

    A Vector is a special matrix with a single column.
    Vectorisation is defined as the process of the linearisation of matices (or
    in this case "columnisation" into a single column), following columns.
    i.e. a Matrix([1, 2], [3, 4]) would be vectorised to Vector(1, 3, 2, 4).
    (check the implementation of the static method Vector.Vectorise for more
    details).
    Item access is identical to Matrix, with the addition of the "Vector-access"
    method via a callable method and a linear 0-index.
    Examples:
       X = matlib.Vector(1, 2, 3, 4)
       X[1] <-> [ 2 ]
       X[:] <-> [ [1], [2], [3], [4] ]
       X[2,0] <-> 3
       X[-1,-1] <-> 4
       X(2) -> 3
       X(-1) -> 4

    Most operations are inherited from the Matrix base class or adapted to
    Vectors. Check individual functions for details.

    Constructor:
        Args:
            *data: Each item as a numerical value, an iterable over numerical
                values or a Matrix. Iterables and Matrix instance are
                "linearised" and added to the matrix (for linearisation details,
                check __init__).
        Examples:
            A = matlib.Matrix([1, 2], [3, 4]) # Matrix to linearise and embed
            X = matlib.Vector(1, 2, (3, 4), [5, 6, 7], A) # All possible combos
    '''
    def __init__(self, *data):
        '''Constructor.'''
        # Validate data
        if not data:
            raise MatrixError("No data provided!")
        # Recursively re-arrange data:
        def linearise(d):
            try:
                # Test if item is iterable, would raise an error if non-iterable
                test = iter(d)
                if isinstance(d, Matrix):
                    # Circulate matrices by columns
                    return linearise([ d[:,j] for j in xrange(d.Columns) ])
                else:
                    # Circulate and join iterables
                    return sum([ linearise(r) for r in d ], [])
            except Exception as e:
                # Non-iterable value wrapped in a list inside a list
                return [ [ d ] ]
        linearised = sum([linearise(datum) for datum in data], [])
        # Form the vector via the Matrix constructor using the linearised data.
        return super(Vector, self).__init__(*linearised)

    def __add__(self, other):
        '''Addition (overrides Matrix.add):

        Vector addition has the following cases:
            % Vector + Vector: straight forward, should return a Vector
            % Vector + scalar: same as in a Matrix, should return a Vector
        '''
        # For non comforming dimensions the super-class __add__ would handle it
        return Vector(super(Vector, self).__add__(other))
    
    def transpose(self):
        '''Transposition (overrides Matrix.transpose):

        Acts directly on self.__matrix__, to avoid extra key calculation
        overhead, by listing all subitems at index 0.
        '''
        return Matrix([ row[0] for row in self.__matrix__ ])
    
    def __mul__(self, other):
        '''Multiplication by a scalar should return a Vector.'''
        if not isinstance(other, Matrix) \
           or other.isScalar or self.Dimension == other.Dimension:
            transform = Vector
        else:
            transform = lambda l: l
        # Validated by the super-class __add__ method, transformed to Vector
        #>for scalars or Vectors.
        return transform(super(Vector, self).__mul__(other))

    def __and__(self, other):
        '''Vertical concatenation should return a Vector'''
        return Vector(super(Vector, self).__and__(other))

    @property
    def ssq(self):
        '''SSQ is the sum of squares of all linearised items for Vectors.'''
        return sum([ r[0]**2 for r in self.__matrix__ ])

    def __invert__(self):
        '''The inverse matrix for a vector is the element-wise division of the
        items of the transposed matrix by the SSQ.'''
        ssq = self.ssq
        return Matrix([ r[0] / ssq for r in self.__matrix__ ])
    
    def apply(self, fn=lambda l: l, *extra_args):
        '''Element-wise custom modifier should return a Vector.'''    
        return Vector(super(Vector, self).apply(fn, *extra_args))
        
class Numtrix(Matrix):
    '''Class returning a matrix containing the same value in all items.

    Constructor:
        num (float): The number with wich to fill the matrix.
        h (int): The height (number of rows) of the matrix.
        w (int): The width (number of columns) of the matrix. Defaults to None,
            resulting in a square matrix.
    '''
    def __init__(self, num, h, w=None):
        super(Numtrix, self).__init__(*[ [num] * (w or h) for i in xrange(h) ])

class Zeroes(Numtrix):
    '''Special subclass of Numtrix containing all zeroes.

    Constructor:
        h (int): The height (number of rows) of the matrix.
        w (int): The width (number of columns) of the matrix. Defaults to None,
            resulting in a square matrix.    
    '''
    def __init__(self, h, w=None):
        super(Zeroes, self).__init__(0.0, h, w)

class Ones(Numtrix):
    '''Special subclass of Numtrix containing all ones.

    Constructor:
        h (int): The height (number of rows) of the matrix.
        w (int): The width (number of columns) of the matrix. Defaults to None,
            resulting in a square matrix.    
    '''
    def __init__(self, h, w=None):
        super(Ones, self).__init__(1.0, h, w)

class Randtrix(Matrix):
    '''Class returning a matrix containing random values:
    
    Constructor:
        h (int): The height (number of rows) of the matrix.
        w (int): The width (number of columns) of the matrix. Defaults to None,
            resulting in a square matrix.
        nrange (float): The maximum value of numbers. Applicable in non-normal-
            distribution mode.
        normal (tuple): Tuple defining the parameters of a normal distribution.
            Defaults to an empty tuple, resulting in pseudo-random (non-normal)
            ditribution.
        modify (callable): Function called on every element of the matrix.
            Defaults to returning the values as they are.
    '''

    def __init__(self, h, w=None, nrange=1, normal=(), modify=lambda l: l):
        data = [ [ modify(random.normalvariate(*normal) if normal
                          else random.random() * nrange)
                   for j in xrange(w or h)] for i in xrange(h) ]
        super(Randtrix, self).__init__(*data)
# Class returning a matrix containing normally distributed random values
class NRandtrix(Randtrix):
    def __init__(self, h, w=None, mu=0, sigma=1):
        super(NRandtrix, self).__init__(h, w, normal=(mu, sigma))

# Identity matrix
class Identity(Matrix):
    def __init__(self, n):
        data = map(lambda i: [0]*i + [1] + [0]*(n-i-1), xrange(n)) if n > 1 \
               else [ [1] ]
        super(Identity, self).__init__(*data)

# Lottery matrix, just for fun
class Lottery(Matrix):
    def __init__(self, n=6, nrange=42):
        # Adjust parameters
        n, nrange = int(n), int(nrange) + 1
        # Initialise
        data, draw = [ b for b in xrange(1, nrange) ], []
        # Draw
        while len(draw) < n:
            # Randomly select an index
            i = int(random.random() * len(data))
            # Draw seleted index
            draw.append(data[i])
            # Remove drawn value
            del data[i]
        # Instantiate
        super(Lottery, self).__init__(draw)

def exp(M):
    return M.apply(math.exp) if isinstance(M, Matrix) else math.exp(M)

def log(M, base=math.e):
    return (M.apply(lambda x: math.log(x, base))
            if isinstance(M, Matrix) else math.log(M, base))

def pow(M, p):
    return (M.apply(lambda x: math.pow(x, p))
            if isinstance(M, Matrix) else math.pow(M, p))
