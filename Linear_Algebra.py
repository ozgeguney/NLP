import numpy as np

alist = [1, 2, 3, 4, 5]
narray = np.array([1, 2, 3, 4])

print(alist)
print(narray)

print(type(alist))
print(type(narray))

print(narray + narray)
print(alist + alist)

print(narray * 3)
print(alist * 3)

npmatrix1 = np.array([narray, narray, narray])
npmatrix2 = np.array([alist, alist, alist])
npmatrix3 = np.array([narray, [1, 1, 1, 1], narray])

print(npmatrix1)
print(npmatrix2)
print(npmatrix3)


okmatrix = np.array([[1, 2], [3, 4]])
print(okmatrix)
print(okmatrix * 2)

result = okmatrix * 2 + 1
print(result)

result1 = okmatrix + okmatrix
print(result1)

result2 = okmatrix - okmatrix
print(result2)

result = okmatrix * okmatrix
print(result)

matrix3x2 = np.array([[1, 2], [3, 4], [5, 6]])
print('Original matrix 3 x 2')
print(matrix3x2)
print('Transposed matrix 2 x 3')
print(matrix3x2.T)

nparray = np.array([1, 2, 3, 4])
print('Original array')
print(nparray)
print('Transposed array')
print(nparray.T)

nparray = np.array([[1, 2, 3, 4]])
print('Original array')
print(nparray)
print('Transposed array')
print(nparray.T)

nparray1 = np.array([1, 2, 3, 4]) # Define an array
norm1 = np.linalg.norm(nparray1)

nparray2 = np.array([[1, 2], [3, 4]])
norm2 = np.linalg.norm(nparray2)

print(norm1)
print(norm2)

nparray2 = np.array([[1, 1], [2, 2], [3, 3]]) # Define a 3 x 2 matrix.

normByCols = np.linalg.norm(nparray2, axis=0) # Get the norm for each column. Returns 2 elements
normByRows = np.linalg.norm(nparray2, axis=1) # get the norm for each row. Returns 3 elements

print(normByCols)
print(normByRows)

nparray1 = np.array([0, 1, 2, 3])  # Define an array
nparray2 = np.array([4, 5, 6, 7])  # Define an array

flavor1 = np.dot(nparray1, nparray2)  # Recommended way
print(flavor1)

flavor2 = np.sum(nparray1 * nparray2)  # Ok way
print(flavor2)

flavor3 = nparray1 @ nparray2  # Geeks way
print(flavor3)

# As you never should do:             # Noobs way
flavor4 = 0
for a, b in zip(nparray1, nparray2):
    flavor4 += a * b

print(flavor4)

#recommend using np.dot, since it is the only method that accepts arrays and lists without problems
norm1 = np.dot(np.array([1, 2]), np.array([3, 4])) # Dot product on nparrays
norm2 = np.dot([1, 2], [3, 4]) # Dot product on python lists

print(norm1, '=', norm2 )

nparray2 = np.array([[1, -1], [2, -2], [3, -3]]) # Define a 3 x 2 matrix.

sumByCols = np.sum(nparray2, axis=0) # Get the sum for each column. Returns 2 elements
sumByRows = np.sum(nparray2, axis=1) # get the sum for each row. Returns 3 elements

print('Sum by columns: ')
print(sumByCols)
print('Sum by rows:')
print(sumByRows)

nparray2 = np.array([[1, -1], [2, -2], [3, -3]]) # Define a 3 x 2 matrix. Chosen to be a matrix with 0 mean

mean = np.mean(nparray2) # Get the mean for the whole matrix
meanByCols = np.mean(nparray2, axis=0) # Get the mean for each column. Returns 2 elements
meanByRows = np.mean(nparray2, axis=1) # get the mean for each row. Returns 3 elements

print('Matrix mean: ')
print(mean)
print('Mean by columns: ')
print(meanByCols)
print('Mean by rows:')
print(meanByRows)

nparray2 = np.array([[1, 1], [2, 2], [3, 3]]) # Define a 3 x 2 matrix.

nparrayCentered = nparray2 - np.mean(nparray2, axis=0) # Remove the mean for each column

print('Original matrix')
print(nparray2)
print('Centered by columns matrix')
print(nparrayCentered)

print('New mean by column')
print(nparrayCentered.mean(axis=0))

nparray2 = np.array([[1, 3], [2, 4], [3, 5]]) # Define a 3 x 2 matrix.

nparrayCentered = nparray2.T - np.mean(nparray2, axis=1) # Remove the mean for each row
nparrayCentered = nparrayCentered.T # Transpose back the result

print('Original matrix')
print(nparray2)
print('Centered by rows matrix')
print(nparrayCentered)

print('New mean by rows')
print(nparrayCentered.mean(axis=1))

nparray2 = np.array([[1, 3], [2, 4], [3, 5]]) # Define a 3 x 2 matrix.

mean1 = np.mean(nparray2) # Static way
mean2 = nparray2.mean()   # Dinamic way
#Even if they are equivalent, we recommend the use of the static way always.
print(mean1, ' == ', mean2)