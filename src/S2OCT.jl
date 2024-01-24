module S2OCT
using Gurobi, JuMP, LinearAlgebra, SCIP
export S2OCTf

##S2OCT is a semi-supervised classification tree proposed by Jan Pablo Burgard, Maria Eduarda Pinheiro, Martin Schmidt avaliable in https://arxiv.org/abs/2401.09848

##S2OCT return the hyperplanes, the objective function and the classification of the unlabeled data

## arguments:
#Xl: Labeled points such that the first ma points belong to class A,
#Xu: Unlabeled points,
## all points belong to R^p
#ma: number of labeled points that belong to class A,
#τ: how many unlabeled points belong to class A,
#D: deep of the tree: integer number between 2 and 5
#C: penalty parameter:  number between 0.5 and 2.
#M: Big M value: η*s*\sqrt{p}+1 where η is the maximum distance between two points in [Xl Xu]
#maxtime: time limit,
#s: bound of ω.
#solver_ By default we use solver=1, which means we are using Gurobi. For that it is necessary a Gurobi license. If choose any different value, SCIP is used.
function S2OCTf(Xl,Xu,ma,τ,D,C,M,maxtime,s,solver=1)
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


end # module

