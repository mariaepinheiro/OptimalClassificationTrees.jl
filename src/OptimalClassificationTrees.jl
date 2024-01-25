module OptimalClassificationTrees

using Gurobi, JuMP, LinearAlgebra, SCIP
export S2OCT
export OCTH
export predictioOCTH
export predictionS2OCT

##S2OCT is a semi-supervised classification tree proposed by Jan Pablo Burgard, Maria Eduarda Pinheiro, Martin Schmidt avaliable in https://arxiv.org/abs/2401.09848

##S2OCT return the hyperplanes, the objective function and the classification of the unlabeled data

## arguments:
#Xl: Labeled points such that the first ma points belong to class A,
#Xu: Unlabeled points,
## all points belong to R^p
#ma: number of labeled points that belong to class A,
#τ: how many unlabeled points belong to class A,
#D: depth of the tree: integer number between 2 and 5
#C: penalty parameter:  number between 0.5 and 2.
#M: Big M value: η*s*\sqrt{p}+1 where η is the maximum distance between two points in [Xl Xu]
#maxtime: time limit,
#s: bound of ω.
#solver_ By default we use solver=1, which means we are using Gurobi. For that it is necessary a Gurobi license. If choose any different value, SCIP is used.
function S2OCT(Xl,Xu,ma,τ,D,C,M,maxtime,s,solver=1)
    ρ = 2^D -1
    p1 = 2^D
    p2 = Int64(p1/2)
    ml,n = size(Xl)
    mu = size(Xu)[1]
    if solver == 1
    model = Model(optimizer_with_attributes(Gurobi.Optimizer))
    set_optimizer_attribute(model, "MIPFocus", 1)
    elseif solver == 0
    model = Model(optimizer_with_attributes(SCIP.Optimizer))
    end 
    set_time_limit_sec(model,maxtime)
    set_silent(model)
   
    @variable(model, -s≤ w[i=1:n,d=1:ρ]≤s)#,start = w1[i,d])
    @variable(model, γ[d=1:ρ])# ,start)= γ1[d])
    @variable(model, α[1:ml,1:p2],Bin)
    @variable(model, 0 ≤ β[1:ml,1:p2])
    @variable(model, yℓ[1:ml,1:ρ] ≥ 0)
    @variable(model, yg[1:ml,1:ρ] ≥ 0)
    @variable(model, zg[1:mu,1:ρ], Bin)
    @variable(model, δ[1:mu,1:p2], Bin)
    @variable(model, ξ≥0)
    @expression(model, zℓ[i=1:mu,j= 1:ρ], -zg[i,j] +1) 
    
    if D ==1 
        AL = [1],zeros(0)
        AR = zeros(0), [1]
    elseif  D == 2
        AL = [1,2],[1],[3],zeros(0)
        AR = zeros(0), [2],[1],[1,3]
    elseif D==3
        AL = [1,2,4], [1,2],[1,5],[1],[3,6],[3],[7],zeros(0)
        AR = zeros(0),[4],  [2], [2,5],[1],[1,6],[1,3],[1,3,7]
    elseif D==4
        AL = [1,2,4,8], [1,2,4],[1,2,9],[1,2],[1,5,10],[1,5],[1,11], [1], [3,6,12], [3,6],[3,13],[3], [7,14], [7],[15], zeros(0)
        AR = zeros(0),  [8],  [4], [4,9],[2],[2,10], [2,5], [2,5,11], [1], [1,12], [1,6], [1,6,13],[1,3],[1,3,14],[1,3,7], [1,3,7,15]
    elseif D==5
        AL = [1,2,4,8,16],[1,2,4,8],[1,2,4,17], [1,2,4], [1,2,9,18], [1,2,9], [1,2,19], [1,2], [1,5,10,20],[1,5,10], [1,5,21], [1,5], [1,11,22],[1,11],[1,23], [1], [3,6,12,24], [3,6,12], [3,6,25], [3,6], [3,13,26],[3,13],[3,27], [3],[7,14,28],[7,14], [7,29], [7], [15,30], [15], [31], zeros(0)
        AR = zeros(0), [16], [8], [8,17], [4], [4,18], [4,9], [4,9,19], [2], [2,20], [2,10], [2,10,21], [2,5], [2,5,22],[2,5,11],[2,5,11,23],[1],[1,24],[1,12],[1,12,25], [1,6],[1,6,26],[1,6,13],[1,6,13,27], [1,3],[1,3,28],[1,3,14],[1,3,14,29], [1,3,7],[1,3,7,30],[1,3,7,15], [1,3,7,15,31]

    end
    @expression(model, LEA[i=1:ma,j=1:p2], sum(yℓ[i,AL[2j-1]])+sum(yg[i,AR[2j-1]]))
    @expression(model, LEB[i=ma+1:ml,j=1:p2], sum(yℓ[i,AL[2j]])+sum(yg[i,AR[2j]]))
    LE = [LEA;LEB]
    for i = 1 : p2
             for k ∈ AL[2i-1]
                @constraint(model, [j=1:mu], δ[j,i] ≤ zℓ[j,k])
            end
            for k ∈ AR[2i-1]
                @constraint(model, [j=1:mu], δ[j,i] ≤ zg[j,k])
            end
        @constraint(model, [j=1:mu], δ[j,i] ≥ sum(zℓ[j,AL[2i-1]]) + sum(zg[j,AR[2i-1]])-(D-1))
    end
   
    @constraint(model, [i=1:ml,d=1:ρ], dot(w[:,d],Xl[i,:]) - γ[d] + 1≤ yℓ[i,d])
    @constraint(model, [i=1:ml,d=1:ρ], -dot(w[:,d],Xl[i,:]) + γ[d]+1 ≤ yg[i,d] )
    @constraint(model, [ix=1:mu,d=1:ρ], dot(w[:,d],Xu[ix,:]) -γ[d] ≤  -1 +zg[ix,d]*M)
    @constraint(model, [ix=1:mu,d=1:ρ], dot(w[:,d],Xu[ix,:]) -γ[d] ≥ 1 -(1-zg[ix,d])*M)
    @constraint(model, [j=1:ml], sum(α[j,:]) == 1)
    @constraint(model, [i=1:ml,j = 1:p2], β[i,j] ≤ (M*D)*α[i,j])
    @constraint(model, [i=1:ml,j = 1:p2], β[i,j] ≤ LE[i,j])
    @constraint(model, [i=1:ml,j = 1:p2], β[i,j] ≥ LE[i,j] - (M*D)*(1-α[i,j]))
    @constraint(model, sum(δ)≤ τ+ξ)
    @constraint(model, sum(δ)≥ τ-ξ)
    @objective(model, Min, sum(β) + C*(sum(ξ)))
    print(model)
    optimize!(model)
    w, γ,fun, δ= value.(w), value.(γ),objective_value(model), value.(δ)
      
    labelclass = -ones(mu) + 2*sum(δ,dims=2)
   
    return w,γ,fun, labelclass
end

##OCTH is a optimal classification tree proposed by Bertsimas and Dunn  avaliable in https://link.springer.com/article/10.1007/s10994-017-5633-9


## arguments:
#X: Labeled points such that the first pos points belong to class A,
#pos number of labeled points that belong to class A,
#D: depth of the tree: integer number between 2 and 5
#α:, . The tradeoff between accuracy and complexity of the tree. Number between 0 and 1
#Nmin: Minimum number of points required in any leaf node,
#maxtime: time limit,
#solver_ By default we use solver=1, which means we are using Gurobi. For that it is necessary a Gurobi license. If choose any different value, SCIP is used.


function OCTH(X,pos,D,α,Nmin, maxtime,solver=1) ###x∈[0,1]
    n,p = size(X)
    TB = 2^D-1
    TL = 2^D
    μ = 0.005
    Y = ones(n,2)
    Y[1:pos,2] = -ones(pos)
    Y[pos+1:n,1] = -ones(n-(pos))
    Y = Y.+1
    hat_L = max(pos, n-pos)
    if solver == 1
    model = Model(optimizer_with_attributes(Gurobi.Optimizer))
    else
    model = Model(optimizer_with_attributes(SCIP.Optimizer))
    end 
    set_time_limit_sec(model,maxtime)
    set_silent(model)
    @variable(model, a[1:TB,1:p])
    @variable(model, hat_a[1:TB,1:p])
    @variable(model, d[1:TB],Bin)
    @variable(model, b[1:TB])
    @variable(model, z[1:TL,1:n],Bin)
    @variable(model, ℓ[1:TL],Bin)
    @variable(model, s[1:TB,1:p],Bin)
    @variable(model, N_k[1:2,1:TL])
    @variable(model, N[1:TL])
    @variable(model, c[1:2,1:TL],Bin)
    @variable(model, L[1:TL]≥0)
    if D == 2
        AL = [1,2],[1],[3],zeros(0)
        AR = zeros(0), [2],[1],[1,3]
    elseif D==3
        AL = [1,2,4], [1,2],[1,5],[1],[3,6],[3],[7],zeros(0)
        AR = zeros(0),[4],  [2], [2,5],[1],[1,6],[1,3],[1,3,7]
    elseif D==4
        AL = [1,2,4,8], [1,2,4],[1,2,9],[1,2],[1,5,10],[1,5],[1,11], [1], [3,6,12], [3,6],[3,13],[3], [7,14], [7],[15], zeros(0)
        AR = zeros(0),  [8],  [4], [4,9],[2],[2,10], [2,5], [2,5,11], [1], [1,12], [1,6], [1,6,13],[1,3],[1,3,14],[1,3,7], [1,3,7,15]
    end
    @constraint(model, [t = 1 : TL, k =1:2], L[t] ≥ N[t]-N_k[k,t] - n*(1-c[k,t]))
    @constraint(model, [t = 1 : TL, k =1:2], L[t] ≤  N[t]-N_k[k,t] + n*c[k,t])
    @constraint(model, [t = 1 : TL, k =1:2], N_k[k,t] == 0.5*(sum(Y[:,k].*z[t,:])))
    @constraint(model, [t = 1 : TL], N[t] == sum(z[t,:]))
    @constraint(model, [t = 1 : TL], ℓ[t] == sum(c[:,t]))
    @constraint(model, [t = 1: TL, m ∈ AL[t], i = 1 : n], dot(a[m,:],X[i,:]) + μ ≤ b[m] + (2+μ)*(1-z[t,i]))
    @constraint(model, [t = 1: TL, m ∈ AR[t], i = 1 : n], dot(a[m,:],X[i,:]) ≥ b[m] -2*(1-z[t,i]))
    @constraint(model, [i=1:n], sum(z[:,i]) == 1)
    @constraint(model, [i=1:n, t = 1:TL], z[t,i]≤ ℓ[t])
    @constraint(model, [t = 1 : TL], sum(z[t,:]) ≥ Nmin*ℓ[t])
    @constraint(model, [t = 1 : TB], sum(hat_a[t,:]) ≤ d[t])
    @constraint(model, [t = 1 : TB, j = 1 : p], hat_a[t,j] ≥ a[t,j])
    @constraint(model, [t = 1 : TB, j = 1 : p], hat_a[t,j] ≥ -a[t,j])
    @constraint(model, [t = 1 : TB, j = 1 : p], a[t,j] ≥ -s[t,j])
    @constraint(model, [t = 1 : TB, j = 1 : p], a[t,j] ≤ s[t,j])
    @constraint(model, [t = 1 : TB, j = 1 : p], s[t,j] ≤ d[t])
    @constraint(model, [t = 1 : TB], b[t] ≤ d[t])
    @constraint(model, [t = 1 : TB], b[t] ≥ -d[t])
    @constraint(model, [t = 2 : TB], d[t] ≤ d[div(t,2)])
    @objective(model, Min, (1/hat_L)*sum(L) + α*sum(s))
    print(model)
    optimize!(model)
    a,b,c,z,fun= value.(a), value.(b), value.(c),value.(z),objective_value(model)
    return a,b,c,z,fun
end

function predictioOCTH(x,a,b,c,D)
    m,p = size(a)
    branch= zeros(m)
    for i = 1 : m 
        if dot(a[i,:],x) - b[i] ≥ 0 
            branch[i] = 1
        end
    end
    if D == 2
        if branch[1] ==0
            if branch[2] == 0
                class = argmax(c[:,1])
            else
                class = argmax(c[:,2])
            end 
        else
            if branch[3] == 0
                class = argmax(c[:,3])
            else
                class = argmax(c[:,4])
            end 
        end
    elseif D == 3
        if branch[1] ==0
            if branch[2] == 0
                if branch[4] == 0
                    class = argmax(c[:,1])
                else
                    class = argmax(c[:,2])
                end 
            else 
                if branch[5] == 0
                    class = argmax(c[:,3])
                else
                    class = argmax(c[:,4])
                end 
            end 
        else
            if branch[3] == 0
                if branch[6] == 0
                    class = argmax(c[:,5])
                else
                    class = argmax(c[:,6])
                end 
            else 
                if branch[7] == 0
                    class = argmax(c[:,7])
                else
                    class = argmax(c[:,8])
                end 
            end 
        end

    elseif D == 4
        if branch[1] ==0
            if branch[2] == 0
                if branch[4] == 0
                    if branch[8] == 0
                        class = argmax(c[:,1])
                    else 
                        class = argmax(c[:,2])
                    end
                else
                    if branch[9] == 0
                        class = argmax(c[:,3])
                    else 
                        class = argmax(c[:,4])
                    end
                end 
            else 
                if branch[5] == 0
                    if branch[10] == 0
                        class = argmax(c[:,5])
                    else
                        class = argmax(c[:,6])
                    end 
                else 
                    if branch[11] == 0
                        class = argmax(c[:,7])
                    else
                        class = argmax(c[:,8])
                    end 
                end 
            end 
        else
            if branch[3] == 0
                if branch[6] == 0
                    if branch[12] == 0
                        class = argmax(c[:,9])
                    else
                        class = argmax(c[:,10])
                    end 
                else
                    if branch[13] == 0
                        class = argmax(c[:,11])
                    else
                        class = argmax(c[:,12])
                    end 
                end 
            else 
                if branch[7] == 0
                    if branch[14] == 0
                        class = argmax(c[:,13])
                    else
                        class = argmax(c[:,14])
                    end 
                else
                    if branch[15] == 0
                        class = argmax(c[:,15])
                    else
                        class = argmax(c[:,16])
                    end 
                end 
            end 
        end
    end
    class = -2*class + 3
    return class

end


function predictionS2OCT(x,w,γ,n=2)
    ac = 0
    if n == 2
        if dot(w[:,1],x)  - γ[1]<0
            if dot(w[:,2],x) - γ[2]< 0
                ac = 1
            else
                ac = -1
            end
        else
            if dot(w[:,3],x) - γ[3]< 0
                ac = 1
            else
                ac = -1
            end
        end
    end 
    if n == 3
        if dot(w[:,1],x)  - γ[1]<0
            if dot(w[:,2],x) - γ[2]< 0
                if dot(w[:,4],x) -γ[4]< 0
                    ac = 1
                else
                    ac = -1
                end
            else 
                if dot(w[:,5],x) -γ[5]< 0
                    ac = 1
                else
                    ac = -1
                end
            end
        else
            if dot(w[:,3],x) - γ[3]< 0
                if dot(w[:,6],x) -γ[6]< 0
                    ac = 1
                else
                    ac = -1
                end
            else
                if dot(w[:,7],x) -γ[7]< 0
                    ac = 1
                else
                    ac = -1
                end   
            end
        end

    end 
    return ac
end

end # module OptimalClassificationTrees
