
#% Numpy introduction

import numpy as np

L = list(range(10)) # Partimos de una lista

A = np.array(L) # La convertimos en un array
type(A), type(L)

# una forma directa es

A = np.arange(10)

# Algunos métodos para operar matematicamente
A.cumsum()
A.sum(), A.prod()
A.min(), A.min(), A.mean()
len(A), A.size

# Replacing elements with put
A.put(7,8) # replace elements: index, values

A.put((0,2,7),(10,12,17))
A.put((1,3),1)
A


# Indexing and slicing

A = np.arange(7)
A[1:4]

A[-3:-1] # backward

A1 = np.arange(100)
A1[-30:-10]

A1[50:100:5] # start, stop, step

A1[::10]

# Vectorazing operation

A1*2

A2 = np.array([A1*2,A1*3])
A2

# Coercing, changing type
A2.astype(float)

## Size, dim and reshape

A3 = np.arange(10)
A4 = np.ones((2,4,2))
A4

# ndim: number of array dimensions
A4.ndim, A3.ndim, A1.ndim 

# size: Number of elements in the array.
A1.size, A3.size, A4.size

# resize: Return a new array with the specified shape.
# If the new array is larger than the original array, then the new array is filled with repeated copies of a

np.resize(A1, (10,11))


# reshape: Gives a new shape to an array without changing its data.

np.arange(6).reshape(3,2) # (row, columns)

A1.reshape(10,10)

A5 = np.arange(200)
A5.ndim, A5.size, A5.shape

A6 = A5.reshape(10,10,2)
A6

# Si queremos volver a un vector unidimensional (aplanar la matriz)

A6.reshape(-1) 
A6.flatten()


# Métodos: concatenate, stack, split, hsplit

A = np.arange(100)
A1 = np.arange(100,200)


# Unamos dos arrays

np.concatenate((A,A1)) #tiene que se un único elemento: lista o tuplas
 
# Función stack

As = np.stack((A,A1)) # Pone una sobre otra, pero de dos dimensiones
As.ndim

# hstack

np.hstack((A,A1)).ndim # No aumenta la dimensión

#dstacks: Stack arrays in sequence depth wise (along third axis).
np.dstack()

# División de arrays

np.split(A1,5)

Asp = np.arange(9)
np.split(Asp,3)
