module NeuralFields

### Gianni Valerio Vinci 2024
using LinearAlgebra,HDF5,SparseArrays,PoissonRandom,DelimitedFiles,QuadGK,SpecialFunctions
BLAS.set_num_threads(1)
""" 
Load Network parameters from a properly formatted h5 file
"""
function LoadNet(fname)

    d=read(h5open(fname))
    SNParam=d["SNParam"]
    CParam=d["CParam"]
    JExt=SNParam["JExt"][:,1]
    KExt=SNParam["NExt"][:,1]
    νExt=SNParam["NuExt"][:,1]
    Δ=SNParam["DeltaExt"][:,1]
    Iext=SNParam["IExt"][:,1]
    N=Int.(SNParam["N"][:,1])
    μx= KExt.*JExt.*νExt +Iext
    σ2x=KExt.*(1 .+Δ.^2).*(JExt.^2).*νExt

    θ=SNParam["Theta"][:,1];
    H=SNParam["H"][:,1];
    g=SNParam["GC"][:,1]
    τC=SNParam["TauC"][:,1];
    τ0=SNParam["Tarp"][:,1];
    τ= SNParam["Beta"][:,1];
    τD= SNParam["TauD"][:,1];
    CJ=CParam["c"].*CParam["J"]
    CJ2=CParam["c"].*(CParam["J"].^2).*(1 .+ CParam["Delta"].^2)

    return N,θ,H,g,τC,τD,τ,τ0,CJ,CJ2,μx,σ2x,CParam["J"]
    
end



function ExponentialAdiacencyMatrix(Ns, λ)
    # Number of total sites
    Npop = Ns^2

    # Create an empty adjacency matrix
    A = Matrix{Float64}(I, Npop, Npop)  # Start with an identity matrix

    # Convert linear indices to 2D coordinates (row, column)
    function index_to_coordinates(idx, Ns)
        row = div(idx - 1, Ns) + 1
        col = rem(idx - 1, Ns) + 1
        return row, col
    end

    # Compute the distance between two points
    function distance(i, j, Ns)
        row_i, col_i = index_to_coordinates(i, Ns)
        row_j, col_j = index_to_coordinates(j, Ns)
        return sqrt((row_i - row_j)^2 + (col_i - col_j)^2)
    end

    # Fill the adjacency matrix with exponential decay weights
    for i in 1:Npop
        for j in i+1:Npop
            dist = distance(i, j, Ns)
            weight = exp(-dist / λ)
            A[i, j] = weight
            A[j, i] = weight  # Symmetric matrix
        end
    end

    return A
end



"""
Initilize Fokker-Planck network from PERSEO initialization files

modulesfile:  PERSEO modules.ini 
connectfile:  PERSEO connectivity.ini
NV: number of grid points in the Fokker-Planck integration
dt: time step
α : regulates the minimum memnrame potential which is set to Vmin=-α*θ 
k : scales the number of neurosn in the nodes.
"""

function InitializeFPfromPerseusNet(modulesfile,connectfile,NV,dt,α,k) 
    d=readdlm(modulesfile)
    N=Int.(d[:,1])
    τm=d[:,6]/1000 # [s]
    θ=d[:,7]       # [mV]
    H=d[:,8]       # [mV]
    τ0=d[:,9]/1000 # [s]
    NExt=d[:,4] 
    νExt=d[:,5]      #[Hz]
    JExt=d[:,2]      #[mV]
    ΔExt=d[:,3]      #[mV]
    τc=d[:,11]/1000  #[s]
    g=d[:,12]*1000
    μx= JExt.*NExt.*νExt  # External mean current
    σ2x=(JExt.^2).*(ΔExt.^2 .+1).*NExt.*νExt # External variance current

    d=readdlm(connectfile)
    Npop=size(N,1)
    c=zeros(Npop,Npop)
    J=zeros(Npop,Npop)
    ΔJ=zeros(Npop,Npop)
    δm=zeros(Npop,Npop)
    δM=zeros(Npop,Npop)

    PostSyn=Int.(d[:,1].+1)
    PretSyn=Int.(d[:,2].+1)

    for i=1:length(PostSyn)
        c[PostSyn[i],PretSyn[i]]=d[i,3]
        J[PostSyn[i],PretSyn[i]]=d[i,7]
        ΔJ[PostSyn[i],PretSyn[i]]=d[i,8]
        δm[PostSyn[i],PretSyn[i]]=d[i,4]/1000
        δM[PostSyn[i],PretSyn[i]]=d[i,5]/1000


    end

    TD=0.05
    τD= (δM.*TD - δm)./(TD - 1) - (δM - δm)/log(TD) # Mean axonal delay [s]


    A,B=zeros(size(c)),zeros(size(c))
    for i=1:size(c)[1]
        for j=1:size(c)[2]
            A[i,j]=c[i,j]*N[j]*J[i,j]
            B[i,j]=c[i,j]*N[j]*(J[i,j]^2)*(1+ΔJ[i,j]^2)
        end
    end
    A=sparse(A)
    B=sparse(B)
    cm=[DefineSinglePopulation(Dict("Vm"=> -α*θ[1],"θ"=>θ[1],"H"=>H[1],"NV"=>NV,
                                "dt"=>dt,"τ"=>τm[1],"τ0"=>τ0[1],"τD"=>τD[1,1],"τC"=>τc[1],"δ"=>δm[1],"g"=>g[1],"N"=>Int(round(k*N[1]))))];
    for n=2:Npop
        push!(cm,DefineSinglePopulation(Dict("Vm"=> -α*θ[n],"θ"=>θ[n],"H"=>H[n],"NV"=>NV,
                                        "dt"=>dt,"τ"=>τm[n],"τ0"=>τ0[n],"τD"=>τD[n,n],"τC"=>τc[n],"δ"=>δm[n,n],"g"=>g[n],"N"=>Int(round(k*N[n])))))
    end

    S=Vector{Float64}[]
    for n=1:Npop
        S=push!(S,InitializeState(cm[n]))
    end
    return cm,S,μx,σ2x,g,Npop,A,B
end

function LoadConnectivity(modulesfile,connectfile)
    d=readdlm(modulesfile)
    N=Int.(d[:,1])
    d=readdlm(connectfile)
    Npop=size(N,1)
    c=zeros(Npop,Npop)
    J=zeros(Npop,Npop)

    PostSyn=Int.(d[:,1].+1)
    PretSyn=Int.(d[:,2].+1)

    for i=1:length(PostSyn)
        c[PostSyn[i],PretSyn[i]]=d[i,3]
        J[PostSyn[i],PretSyn[i]]=d[i,7]
    end

    K=zeros(Npop,Npop)
    for i=1:Npop 
        for j=1:Npop 
            K[i,j]=c[i,j]*N[j]
        end
    end
    
    return K,J
end


function BuildTile(gt,ϵ,N,λ,γ,L,NV,dt,α)
    #Cortical columns params
    Scale=N/6300
    cm,S,μx,σ2x,g,Npop,KJ,KJ2=InitializeFPfromPerseusNet("mod.ini","con.ini",NV,dt,α,Scale);
    g[1]=gt
    g[2]=0.0
    μx[1]=μx[1]*ϵ
    σ2x[1]=σ2x[1]*(ϵ^2)
    Jcc=KJ2[1,1]/KJ[1,1]
    Kcc=KJ[1,1]/Jcc
    
    Npop=L^2
    #Initialize State:
    SE=Vector{Float64}[]
    SI=Vector{Float64}[]

    for n=1:Npop
        SE=push!(SE,copy(S[1]))
        SI=push!(SI,copy(S[2]))
    end

    K=ExponentialAdiacencyMatrix(L, λ)
    K[findall(x-> x<0.05,K)].=0 # Pruning
    K[diagind(K)].=0
    K=K.*Kcc
    
    KJF=sparse(K.*(Jcc*γ))
    KJ2F=sparse(K.*((Jcc*γ).^2))
    return cm,SE,SI,μx,σ2x,g,Npop,KJ,KJ2,KJF,KJ2F
end


const sqπ=sqrt(π)
const tol=1e-8;
const tt=5


## Function required for computing the Stationary firing rate of LIF SNN

function G(w)
	if(w>=-3)
		return exp(w^2)*(erf(w)+1)
	else
		return exp(w^2)*(erf(w)+erf(tt-w))
    end
end

MFPTfrom0(b)= sqπ*dawson(b)*G(b) - 2*quadgk(x -> dawson(x), 0, b, rtol=tol)[1]


function MFPT2(a,b)
	return MFPTfrom0(b)-MFPTfrom0(a)
end

function MFPT3(a,b)
	return quadgk(w ->  exp(-(w^2))*(exp(2*b*w)-exp(2*a*w))/w, 0, Inf, rtol=tol)[1]
end


function MFPT(a,b)
	if(b<=0)
		return MFPT3(a,b)

	else
		if(b>15)
			return Inf

		else
			if(a<=0)
				return MFPT3(a,0)+MFPT2(0,b)

			else
				return MFPT2(a,b)
			end
		end
	end
end

#LIF gain function
function Φ(μ,σ,θ,H,τ,τ₀)
	xr,xt=(H-μ)/σ,(θ-μ)/σ
	return 1/(τ*MFPT(xr,xt) +τ₀)

end


## Fokker-Planck modules

"""
SinglePopulation is a const. structure that contains all the fixed 
parameters of the single node Fokker-Planck equations
"""

struct SinglePopulation
    #Integration time step
    dt::Float64
    #Membrane decay time
    τ::Float64
    #Absolute refractory period
    τ0::Float64
    #Mean delay time
    τD::Float64
    #Adaptation time scale
    τC::Float64
    #Threshold and reset potential
    θ::Float64
    H::Float64
    dV::Float64
    V::Vector{Float64}
    Vc::Vector{Float64}
    NV::Int
    ib::Int
    nref::Int
    nδ::Int
    g::Float64
    N::Int

end

"""
Initilize the grid and intial condition with
p(v,t=0)=δ(v-H)
"""
function GridInitialization(Vmin,H,θ,NV)

    VC=Array(range(Vmin,stop=θ,length=NV))
    dVCenters=(VC[end]-VC[end-1])/2;

    VC=VC .- dVCenters;
    dVInterfaces=diff(VC);
    VI=zeros(NV+1)

    VI[2:end-1]=VC[1:NV-1] + (0.5).*dVInterfaces
    VI[end]=VC[end] + (0.5).* dVInterfaces[end];
    VC[1]=VC[1] - (0.5).* dVInterfaces[1];
    dV=VC[3]-VC[2]
    ib=argmin(abs.(VC.-H)) #reinjection index
    return VI,VC,ib,dV
end


"""
Define the single population from a Dict of parameters p
"""
function DefineSinglePopulation(p)
    Vmin=p["Vm"]
    NV=p["NV"]
    V,Vc,ib,dV=GridInitialization(Vmin,p["H"],p["θ"],NV)

    nref=Int(round(p["τ0"]/p["dt"]))
    nδ=Int(round(p["δ"]/p["dt"]))


    cm=SinglePopulation( p["dt"],p["τ"],p["τ0"],p["τD"],
                         p["τC"],p["θ"],p["H"],dV,V,Vc,NV,
                         ib,nref,nδ,p["g"],p["N"])
    return cm
end

"""
Initialize a vector that will contain all the state variable of the single nodes
"""
function InitializeState(cm)
    S=zeros(cm.NV +3 +max(cm.nref,cm.nδ))
    #Define State
    S[cm.ib]=1/cm.dV  # default initialization is p=δ(v-H)
    S[cm.NV+1]=0.0    #  ν
    S[cm.NV+2]=0.0    # νd
    S[cm.NV+3]=0.0    # c
    return S
end

function exp_vdV_D(v,dV,D)
    return exp(-v*dV/D)
end


# Deterministic component of LIF neural model
function GetF!(f,V,μ,τ,El,Nv)
    @inbounds for i=1:Nv
        f[i]=(El-V[i])/τ +μ
    end
end


# Diagonals of matrix rapresentation of the Fokker-Planck evolution operator:
function Diagonals!(mat,N,v,D,dV,dt)
    dt_dV = dt/dV
    @inbounds for i=2:N-1
        if (v[i] != 0.0)
            exp_vdV_D1 = exp_vdV_D(v[i],dV,D)
            mat[2,i] = -dt_dV*v[i]*exp_vdV_D1/(1.0 -exp_vdV_D1) # diagonal
            mat[3,i-1] = dt_dV*v[i]/(1.0 -exp_vdV_D1) # lower diagonal
        else
            mat[1,i] = -dt_dV*D/dV # diagonal
            mat[2,i-1] = dt_dV*D/dV # lower diagonal
        end

        if (v[i+1]!=0.0)
            exp_vdV_D2 = exp_vdV_D(v[i+1],dV,D)
            mat[2,i] -= dt_dV*v[i+1]/(1.0-exp_vdV_D2) # diagonal
            mat[1,i+1] = dt_dV*v[i+1]*exp_vdV_D2/(1.0 -exp_vdV_D2) # upper diagonal
        else
            mat[2,i] -= dt_dV*D/dV # diagonal
            mat[1,i+1] = dt_dV*D/dV # upper diagonal
        end
    end

    # Boundary conditions
    if (v[2] != 0.0)
        tmp1 = v[2]/(1.0 -exp_vdV_D(v[2],dV,D))
    else
        tmp1 = D/dV
    end

    if (v[end] != 0.0)
        tmp2 = v[end]/(1.0 -exp_vdV_D(v[end],dV,D))
    else
        tmp2 = D/dV
    end
    if (v[end-1] != 0.0)
        tmp3 = v[end-1]/(1.0 -exp_vdV_D(v[end-1],dV,D))
    else
        tmp3 = D/dV
    end

    mat[2,1] = -dt_dV*tmp1                      # first diagonal
    mat[1,2] = dt_dV*tmp1*exp_vdV_D(v[2],dV,D)  # first upper
    mat[3,end-1] = dt_dV*tmp3                   # last lower
    mat[2,end] = -dt_dV * ( tmp3*exp_vdV_D(v[end-1],dV,D)
                          +tmp2*(1.0 +exp_vdV_D(v[end],dV,D)) )  # last diagonal
    return

end;

"""
1 step integration of the single node Fokker-Planck
pop: is the single population struc. of parameters
μ: mean of the total synaptic current of the node
σ2: variance of the total synaptic current of the node
S: state vector where the evolution is stored
FS: Bolean if true implements finite-size fluctuations
"""
function IntegrateFP!(pop::SinglePopulation,μ::Float64, σ2::Float64,S::Vector{Float64},FS::Bool)

    #Unpack the state
    NV=pop.NV
    p=@view S[1:NV]
    νH=@view S[NV+4:end] # History of firning rate
    ν,νd,c=S[NV+1],S[NV+2],S[NV+3]

    #Consider to include Adt in the State
    Adt=zeros(3,NV)
    v=zeros(NV+1)


    # Number of realizations in refractory period
    IntRef=sum(νH[end-pop.nref+1:end])*pop.dt
    # calculate mass inside and outside the comp. domain
    IntP = sum(p)*pop.dV
    # normalize the probability distribution
    p*=(1.0-IntRef)/IntP

    # Fokker-Planck Drift
    GetF!(v,pop.V,μ,pop.τ,0.0,pop.NV+1)
    # Fokker-Planck Diffusion
    D = 0.5*σ2
    #Update Adt bandend matrix
    Diagonals!(Adt,pop.NV,v,D,pop.dV,pop.dt)
    #Reinjecton of flux in H:
    p[pop.ib] += νH[end-pop.nref+1]*(pop.dt/pop.dV)  # reinjection of ν(t-τ₀)
    Adt *= -1
    Adt[2,:] += ones(pop.NV)

    # solve the linear system
    p[1:end].=LAPACK.gtsv!(Adt[3,1:end-1],Adt[2,:], Adt[1,2:end], p)

    if v[end] != 0.0
        ν = v[end]*((1.0+exp((-v[end]*pop.dV)/D))/(1.0-exp((-v[end]*pop.dV)/D)))*p[end]
    else
        ν = 2*D/pop.dV * p[end]
    end

    #Update firing rate history
    if FS
        νN=pois_rand(ν*pop.N*pop.dt)/(pop.N*pop.dt) # !!!Search for faster implementation
    else
        νN=ν
    end

    νH[1:end-1]=νH[2:end]
    νH[end]=νN 

    νd = νd + pop.dt *(νH[end-pop.nδ+1]- νd)/pop.τD
    c=   c  + pop.dt*( -c/pop.τC +ν)


    #Update
    S[1:NV]=p
    S[NV+1],S[NV+2],S[NV+3]=ν,νd,c
    return nothing
end;



end