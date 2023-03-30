from scipy.fft import fft, ifft
import numpy as np
from scipy import signal
from scipy import linalg
import statsmodels.api as sm

# Add n-1 zeros at the end of vector x of size n
def doublePadRight(x): return np.pad(x, (0, x.size - 1), 'constant', constant_values=(0,))

# Add n-1 zeros at the beginning of vector x of size n
def doublePadLeft(x): return np.pad(x, (x.size - 1, 0), 'constant', constant_values=(0,))

# Creates an nxn diagonal matrix from vector x of size n
#def matrixFromDiagonal(x): return np.multiply(np.identity(x.size),x)

# Multiply a Toeplitz matrix, with first column col and first row zeros, with matrix m
def mulToeplitzFromCol(col, rowSize, matrix):
    return linalg.matmul_toeplitz((col, np.zeros(rowSize)), matrix)

# Multiply a Toeplitz matrix, with first row row and first column zeros, with matrix m
def mulToeplitzFromRow(row, colSize, matrix):
    return linalg.matmul_toeplitz((np.concatenate((row[0:1], np.zeros(colSize-1))), row), matrix)

# Create Toeplitz matrix with first column c and first row zeros
#def toeplizFromRow(x): return mulToeplitzFromRow(x, np.identity(x.size))

# Create Toeplitz matrix with first column c and first row zeros
#def toeplizFromCol(x): return mulToeplitzFromRow(x, np.identity(x.size))

def autocorrByToeplitzMul(seq, maxDist, oneSide = True):
#    return mulToeplitzFromRow(doublePadRight(x),np.diag(doublePadLeft(x)))
    rightPaddedSeq = np.pad(seq, (0, maxDist), 'constant', constant_values=(0,))
    leftPaddedSeq = np.pad(seq, (maxDist, 0), 'constant', constant_values=(0,))
    xMask = np.delete(np.diag(leftPaddedSeq), range(0, maxDist), axis = 1)
    # return mulToeplitzFromRow(doublePadRight(x), x.size, xMask)[x.size - 1 - maxDist:x.size + maxDist]
    if oneSide:
        return np.flip(mulToeplitzFromRow(rightPaddedSeq, maxDist + 1, xMask), axis = 0)
    else:
        return mulToeplitzFromRow(rightPaddedSeq, 2 * maxDist + 1, xMask)
    
def autocorr(seq, maxDist, oneSide = True):
    return sm.tsa.acf(seq, nlags = maxDist)

# def repeatsByPositionAndDistance(seqStr,maxDist):
#     seqByBase = {base: np.array([1 if c==base else 0 for c in seqStr]) for base in ['A', 'C', 'G', 'T']}
#     convByBase = {base: conv(seqByBase[base],maxDist) for base in "ACGT"}
#     totalConv = convByBase['A'] + convByBase['C'] + convByBase['G'] + convByBase['T']
#     return totalConv

def repeatsByPositionAndDistance(seqStr, maxDist, oneSide = True, absolute = True):
    seqByBase = {base: np.array([1 if c==base else 0 for c in seqStr]) for base in ['A', 'C', 'G', 'T']}
    convByBase = {base: autocorrByToeplitzMul(seqByBase[base], maxDist, oneSide) for base in "ACGT"}
    repeats = convByBase['A'] + convByBase['C'] + convByBase['G'] + convByBase['T']
    # repeats = repeatsByPositionAndDistance(seqStr,maxDist)
    if (not oneSide and absolute):
        seqSize = repeats.shape[1]
        a = np.array([repeats[maxDist:],np.vstack((np.ones(seqSize), np.flip(repeats[0:maxDist], axis=0)))])
        repeats = np.average(a,axis=0)
        matrixOfOnes = np.ones(repeats.shape)
        normalizer = matrixOfOnes + np.tril(matrixOfOnes, -1) + np.flip(np.triu(matrixOfOnes, seqSize - maxDist), axis = 0)
        repeats = repeats * normalizer
    #    return normalizer
    #    lowerTriangle = + np.tril(matrixOfOnes, -maxDist-1)
    # convRadius = maxDist
    # convMatrix = np.asmatrix(np.ones(2 * convRadius - 1)/(2 * convRadius - 1))
    # repeats = signal.convolve2d(repeats,convMatrix)
    return repeats

def uniformSmooth(seq, convRadius):
    convMatrix = np.asmatrix(np.ones(2 * convRadius - 1)/(2 * convRadius - 1))
    return signal.convolve2d(seq, convMatrix, boundary = 'symm')

def linearSmooth(seq, convRadius):
    convMatrix = np.asmatrix(np.concatenate((np.arange(1,convRadius + 1), np.arange(convRadius - 1,0,-1)))/(convRadius*convRadius))
    return signal.convolve2d(seq, convMatrix, boundary = 'symm')

def totRepeatRatiosByDistance(seqStr):
    totalConv = repeatsByPositionAndDistance(seqStr)
    seqSize = int(totalConv.shape[0]/2) + 1
    maxValues = np.concatenate((np.arange(1,seqSize + 1),np.arange(seqSize - 1,0,-1)))
    return totalConv.sum(axis=1)/maxValues

