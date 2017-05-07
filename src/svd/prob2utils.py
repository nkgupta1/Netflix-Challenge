'''
matrix factorization
'''

import sys
import numpy as np

def grad_U(Ui, Yij, Vj, ai, bj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return (1-reg*eta)*Ui + eta * Vj * (Yij - (np.dot(Ui,Vj) + ai + bj))

def grad_V(Vj, Yij, Ui, ai, bj, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return (1-reg*eta)*Vj + eta * Ui * (Yij - (np.dot(Ui,Vj) + ai + bj))

def grad_a(Ui, Yij, Vj, ai, bj, reg, eta):
    return (1-reg*eta)*ai + eta * (Yij - (np.dot(Ui,Vj) + ai + bj))

def grad_b(Ui, Yij, Vj, ai, bj, reg, eta):
    return (1-reg*eta)*bj + eta * (Yij - (np.dot(Ui,Vj) + ai + bj))

def get_err(U, V, a, b, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V.
    """
    # Compute mean squared error on each data point in Y; include
    # regularization penalty in error calculations.
    # We first compute the total squared squared error
    err = 0.0
    for (i,j,Yij) in Y:
        i, j = int(i), int(j)
        err += 0.5 *(Yij - (np.dot(U[i-1], V[:,j-1]) + a[i-1] + b[j-1]))**2
    # Add error penalty due to regularization if regularization
    # parameter is nonzero
    if reg != 0:
        U_frobenius_norm = np.linalg.norm(U, ord='fro')
        V_frobenius_norm = np.linalg.norm(V, ord='fro')
        a_frobenius_norm = np.dot(a, a)
        b_frobenius_norm = np.dot(b, b)
        err += 0.5 * reg * (U_frobenius_norm ** 2)
        err += 0.5 * reg * (V_frobenius_norm ** 2)
        err += 0.5 * reg * (a_frobenius_norm)
        err += 0.5 * reg * (b_frobenius_norm)
    # Return the mean of the regularized error
    return err / float(len(Y))

def train_model(M, N, K, eta, reg, Y, Y_test=None, eps=0.0001, max_epochs=300):
    """
    Factorizes the M x N matrix Y into the product of an M x K matrix
    U and a K x N matrix V (i.e. Y = UV).

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    # Initialize U, V
    U = np.random.random((M,K)) - 0.5
    V = np.random.random((K,N)) - 0.5
    a = np.random.random((M))   - 0.5
    b = np.random.random((N))   - 0.5

    size = Y.shape[0]
    delta = None
    indices = list(range(size))
    for epoch in range(max_epochs):
        # Run an epoch of SGD
        before_E_in = get_err(U, V, a, b, Y, reg)
        np.random.shuffle(indices)
        for ind in indices:
            (i,j, Yij) = Y[ind]
            i,j = int(i),int(j)
            # Update U[i], V[j], a[i], b[i]
            U[i-1] = grad_U(U[i-1], Yij, V[:,j-1], a[i-1], b[j-1], reg, eta)
            V[:,j-1] = grad_V(V[:,j-1], Yij, U[i-1], a[i-1], b[j-1], reg, eta)
            a[i-1] = grad_a(U[i-1], Yij, V[:,j-1], a[i-1], b[j-1], reg, eta)
            b[j-1] = grad_b(U[i-1], Yij, V[:,j-1], a[i-1], b[j-1], reg, eta)
        # At end of epoch, print E_in
        E_in = get_err(U, V, a, b, Y, reg)
        out_str = 'Epoch {:3d}, E_in: {:.10f}'.format(epoch+1, E_in)
        if Y_test is not None:
            E_out = get_err(U, V, a, b, Y_test, reg)
            out_str = '{}, E_out: {:.10f}'.format(out_str, E_out)
        print(out_str)

        # Compute change in E_in for first epoch
        if epoch == 0:
            delta = before_E_in - E_in

        # If E_in doesn't decrease by some fraction <eps>
        # of the initial decrease in E_in, stop early            
        elif before_E_in - E_in < eps * delta:
            break
    return (U, V, a, b, get_err(U, V, a, b, Y))
    