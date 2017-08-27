import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
 
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores=X.dot(W)
  for x in range(X.shape[0]):
    scores[x,:]-=np.amax(scores[x,:])
    loss+=-scores[x,y[x]]+np.log(np.sum(np.exp(scores[x,:])))
    
    for i in range(W.shape[1]):
        temp=np.exp(scores[x,i])/np.sum(np.exp(scores[x,:]))
        if i == y[x]:
            dW[:,i]+=(-1+temp)*X[x,:]
        else:
            dW[:,i]+=temp*X[x,:]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss=loss/X.shape[0]+0.5*reg*np.sum(W*W)
  dW=dW/X.shape[0]+reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores=X.dot(W)
  scores-=np.amax(scores)
  loss=np.sum(-scores[np.arange(X.shape[0]),y]+np.log(np.sum(np.exp(scores),axis=1)),axis=0)/X.shape[0]
  loss+=0.5*reg*np.sum(W*W)  
    
  coeff_W=np.zeros((X.shape[0],W.shape[1]))
  base=np.sum(np.exp(scores),axis=1)
  base=base[:,np.newaxis]
  coeff_W+=np.exp(scores)/base
  coeff_W[np.arange(X.shape[0]),y]-=1
  dW=(X.T.dot(coeff_W))/X.shape[0]
  dW+=reg*W
    
    
    
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

