# OptimalClassificationTree.jl

This package provides the semi-supervised decision tree proposed in [Mixed-Integer Linear Optimization for Semi-Supervised Optimal Classification Trees](https://arxiv.org/abs/2401.09848) authored by Jan Pablo Burgard, Maria Eduarda Pinheiro, Martin Schmidt.

It also provides the optimal classification tree proposed in [Optimal Classification Trees](https://link.springer.com/article/10.1007/s10994-017-5633-9) proposed by   Dimitris Bertsimas and Jack Dunn.


## Take a very simple example

- Xl= [0.636331    0.409577   0.0343776  0.49033   0.941649
 0.0988082   0.63994    0.841382   0.421412  0.235375
 0.995567    0.0274223  0.992937   0.458846  0.199137
 0.199072    0.691256   0.580256   0.775544  0.526364
 0.846138    0.747581   0.284312   0.338553  0.486645
 0.00898094  0.975308   0.103471   0.852332  0.258365
 0.301501    0.0219966  0.511066   0.275679  0.0726101
 0.691839    0.171178   0.230506   0.66391   0.224102
 0.933951    0.626902   0.422172   0.473584  0.986075
 0.029338    0.433094   0.516192   0.303025  0.191949]

- Xu = [0.354165   0.205495  0.826024  0.760528  0.438385
 0.548362   0.719404  0.641118  0.012956  0.941432
 0.139877   0.293286  0.495163  0.273754  0.849356
 0.30901    0.434707  0.135739  0.88631   0.554937
 0.70241    0.884161  0.293552  0.933542  0.00816189
 0.0677852  0.14013   0.16357   0.169234  0.818138
 0.0798973  0.597091  0.253266  0.212307  0.725866
 0.03988    0.389887  0.842093  0.527728  0.527742
 0.505142   0.592454  0.952136  0.530725  0.0462085
 0.337482   0.367811  0.9618    0.267502  0.906256
 0.338645   0.134139  0.495685  0.928878  0.136617
 0.21697    0.465524  0.776617  0.714563  0.337095
 0.300243   0.584082  0.774677  0.600761  0.277836
 0.825492   0.865387  0.254526  0.667541  0.653019
 0.55255    0.693362  0.771232  0.166322  0.619182]

- ma = 5

- pos = 5

- τ = 5

- C = 1

- D = 2

- M = 100

- s= 10

- solver = 0

- maxtime = 3600

- α = 0

- Nmin = 1

- out1 =  S2OCT(Xl,Xu,ma,τ ,C,M,maxtime,s,solver) 


- out2 = OCTH(Xl,pos,D,,α,Nmin, maxtime,solver)
