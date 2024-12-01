using Plots
using HDF5
using Polyester

const σx = 0.1#Don't underresolve this for the love of god. 
const σt = 0.1
const L = 5.0
const dx = 0.0125
const dt = 0.00625
const Nx  = 400
const Ntp = 400
const Nt  = 400
const cfl = dt/dx
const cfl2 = cfl^2
const vac_modes = 100

Wightman_vac(x::Real,y::Real) = (1/pi)*sum(((1/N)*exp(-(N^2*pi^2*σx^2)/L^2)*exp(-(N^2*pi^2*σt^2)/L^2)*
                        sinpi((N*x)/L)*sinpi((N*y)/L)) for N in 1:vac_modes)  

∂tWightman_vac(x::Real,y::Real) = (-im/L)*sum((exp(-(N^2*pi^2*σx^2)/L^2)*exp(-(N^2*pi^2*σt^2)/L^2)*
                        sinpi((N*x)/L)*sinpi((N*y)/L)) for N in 1:vac_modes)

∂τWightman_vac(x::Real,y::Real) = (im/L)*sum((exp(-(N^2*pi^2*σx^2)/L^2)*exp(-(N^2*pi^2*σt^2)/L^2)*
                        sinpi((N*x)/L)*sinpi((N*y)/L)) for N in 1:vac_modes)

∂t∂τWightman_vac(x::Real,y::Real) = (pi/(L^2))*sum(N*(exp(-(N^2*pi^2*σx^2)/L^2)*exp(-(N^2*pi^2*σt^2)/L^2)*
                        sinpi((N*x)/L)*sinpi((N*y)/L)) for N in 1:vac_modes)

function IC1!(slab_read::AbstractArray{<:Complex,4},x::Vector{Float64})
    @batch for j in 1:Nx+1, i in 1:Nx+1
        slab_read[i,j,1,1] = Wightman_vac(x[i],x[j])
    end
    return

end

function IC2!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4},x::Vector{Float64})
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,1,2] = slab_read[i,j,1,1] + (1/2)*cfl2*(slab_read[i,j+1,1,1] - 2*slab_read[i,j,1,1] + slab_read[i,j-1,1,1]) + dt*∂τWightman_vac(x[i],x[j])

        slab_write[i,1,1,2] = 0.0
        slab_write[i,end,1,2] = 0.0
        slab_write[1,j,1,2] = 0.0
        slab_write[end,j,1,2] = 0.0
    end
    return
end

function IC3!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4},x::Vector{Float64})
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,2,1] = slab_read[i,j,1,1] + (1/2)*cfl2*(slab_read[i+1,j,1,1] - 2*slab_read[i,j,1,1] + slab_read[i-1,j,1,1]) + dt*∂tWightman_vac(x[i],x[j])

        slab_write[i,1,2,1] = 0.0
        slab_write[i,end,2,1] = 0.0
        slab_write[1,j,2,1] = 0.0
        slab_write[end,j,2,1] = 0.0
    end
    return
end

function IC4!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4},x::Vector{Float64})
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,2,2] = slab_read[i,j,1,2] + slab_read[i,j,2,1] - slab_read[i,j,1,1] + dt^2*∂t∂τWightman_vac(x[i],x[j])

        slab_write[1,j,2,2] = 0.0
        slab_write[end,j,2,2] = 0.0
        slab_write[i,1,2,2] = 0.0
        slab_write[i,end,2,2] = 0.0
    end
    return
end

function Update_IC!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4})

    for j in 1:Nx+1, i in 1:Nx+1 
        slab_read[i,j,1,2] = slab_write[i,j,1,2]
        slab_read[i,j,2,1] = slab_write[i,j,2,1]
        slab_read[i,j,2,2] = slab_write[i,j,2,2]
    end
    return
end

function Apply_BCs!(slab_write::AbstractArray{<:Complex,4},n::Int64,m::Int64)

    @batch for j in 2:Nx, i in 2:Nx
        slab_write[1,j,n,m] = 0.0
        slab_write[end,j,n,m] = 0.0
        slab_write[i,1,n,m] = 0.0
        slab_write[i,end,n,m] = 0.0
    end
    return
end

function Write_to_Read!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4},n::Int64,m::Int64)
    @batch for j in 1:Nx+1, i in 1:Nx+1
        slab_read[i,j,n,m] = slab_write[i,j,n,m]
    end

    return
end

function Make_Slab!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4})
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,1,3] = 2*slab_read[i,j,1,2] - slab_read[i,j,1,1] + cfl2*(slab_read[i,j+1,1,2] - 2*slab_read[i,j,1,2] + slab_read[i,j-1,1,2])
        slab_write[i,j,2,3] = 2*slab_read[i,j,2,2] - slab_read[i,j,2,1] + cfl2*(slab_read[i,j+1,2,2] - 2*slab_read[i,j,2,2] + slab_read[i,j-1,2,2])
    end

    Apply_BCs!(slab_write,1,3)
    Apply_BCs!(slab_write,2,3)

    Write_to_Read!(slab_write,slab_read,1,3)
    Write_to_Read!(slab_write,slab_read,2,3)

    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,3,1] = 2*slab_read[i,j,2,1] - slab_read[i,j,1,1] + cfl2*(slab_read[i+1,j,2,1] - 2*slab_read[i,j,2,1] + slab_read[i-1,j,2,1])
        slab_write[i,j,3,2] = 2*slab_read[i,j,2,2] - slab_read[i,j,1,2] + cfl2*(slab_read[i+1,j,2,2] - 2*slab_read[i,j,2,2] + slab_read[i-1,j,2,2])
        slab_write[i,j,3,3] = 2*slab_read[i,j,2,3] - slab_read[i,j,1,3] + cfl2*(slab_read[i+1,j,2,3] - 2*slab_read[i,j,2,3] + slab_read[i-1,j,2,3])
    end

    Apply_BCs!(slab_write,3,1)
    Apply_BCs!(slab_write,3,2)
    Apply_BCs!(slab_write,3,3)

    Write_to_Read!(slab_write,slab_read,3,1)
    Write_to_Read!(slab_write,slab_read,3,2)
    Write_to_Read!(slab_write,slab_read,3,3)

    return
end

function Evo_Slab_Up!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4},m::Int64)
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,1,3] = 2*slab_read[i,j,1,3] - slab_read[i,j,1,2] + cfl2*(slab_read[i,j+1,1,3] - 2*slab_read[i,j,1,3] + slab_read[i,j-1,1,3])
        slab_write[i,j,2,3] = 2*slab_read[i,j,2,3] - slab_read[i,j,2,2] + cfl2*(slab_read[i,j+1,2,3] - 2*slab_read[i,j,2,3] + slab_read[i,j-1,2,3])
        slab_write[i,j,3,3] = 2*slab_read[i,j,3,3] - slab_read[i,j,3,2] + cfl2*(slab_read[i,j+1,3,3] - 2*slab_read[i,j,3,3] + slab_read[i,j-1,3,3])
    end

    Apply_BCs!(slab_write,1,3)
    Apply_BCs!(slab_write,2,3)
    Apply_BCs!(slab_write,3,3)

    for j in 1:Nx+1, i in 1:Nx+1
        slab_read[i,j,1,1] = slab_read[i,j,1,2]
        slab_read[i,j,2,1] = slab_read[i,j,2,2]
        slab_read[i,j,3,1] = slab_read[i,j,3,2]

        slab_read[i,j,1,2] = slab_read[i,j,1,3]
        slab_read[i,j,2,2] = slab_read[i,j,2,3]
        slab_read[i,j,3,2] = slab_read[i,j,3,3]

        slab_read[i,j,1,3] = slab_write[i,j,1,3]
        slab_read[i,j,2,3] = slab_write[i,j,2,3]
        slab_read[i,j,3,3] = slab_write[i,j,3,3]
    end

    return
end

function Evo_Slab_Right!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4},m::Int64)
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,3,1] = 2*slab_read[i,j,3,1] - slab_read[i,j,2,1] + cfl2*(slab_read[i+1,j,3,1] - 2*slab_read[i,j,3,1] + slab_read[i-1,j,3,1])
        slab_write[i,j,3,2] = 2*slab_read[i,j,3,2] - slab_read[i,j,2,2] + cfl2*(slab_read[i+1,j,3,2] - 2*slab_read[i,j,3,2] + slab_read[i-1,j,3,2])
        slab_write[i,j,3,3] = 2*slab_read[i,j,3,3] - slab_read[i,j,2,3] + cfl2*(slab_read[i+1,j,3,3] - 2*slab_read[i,j,3,3] + slab_read[i-1,j,3,3])
    end

    Apply_BCs!(slab_write,3,1)
    Apply_BCs!(slab_write,3,2)
    Apply_BCs!(slab_write,3,3)

    for j in 1:Nx+1, i in 1:Nx+1 
        slab_read[i,j,1,1] = slab_read[i,j,2,1]
        slab_read[i,j,1,2] = slab_read[i,j,2,2]
        slab_read[i,j,1,3] = slab_read[i,j,2,3]

        slab_read[i,j,2,1] = slab_read[i,j,3,1]
        slab_read[i,j,2,2] = slab_read[i,j,3,2]
        slab_read[i,j,2,3] = slab_read[i,j,3,3]

        slab_read[i,j,3,1] = slab_write[i,j,3,1]
        slab_read[i,j,3,2] = slab_write[i,j,3,2]
        slab_read[i,j,3,3] = slab_write[i,j,3,3]
    end
    return
end

function Evo_to_tprime!(slab_write::AbstractArray{<:Complex,4}, slab_read::AbstractArray{<:Complex,4})

    for m in 4:Ntp+1
        Evo_Slab_Up!(slab_write,slab_read,m)
        println("m = $m")
    end
    return
end

function Solve_Wightman()

    x = [0.0 + i*dx for i in 0:Nx]
    t = [0.0 + i*dt for i in 0:Nt]
    println(t[end])
    W_storage = Array{ComplexF64}(undef, Nx+1, Nx+1, Nt+1)

    slab_read = Array{ComplexF64}(undef,Nx+1,Nx+1,3,3)
    slab_write = Array{ComplexF64}(undef,Nx+1,Nx+1,3,3)

    IC1!(slab_read,x)
    IC2!(slab_write,slab_read,x)
    IC3!(slab_write,slab_read,x)
    Update_IC!(slab_write,slab_read)
    IC4!(slab_write,slab_read,x)
    Update_IC!(slab_write,slab_read)
    Make_Slab!(slab_write,slab_read)

    heatmap(x,x,real.(slab_read[:,:,1,1]))

    Evo_to_tprime!(slab_write,slab_read)

    for j in 1:Nx+1, i in 1:Nx+1
        W_storage[i,j,1] = slab_read[i,j,1,1]
        W_storage[i,j,2] = slab_read[i,j,2,3]
        W_storage[i,j,3] = slab_read[i,j,3,3]
    end
    
    for n in 4:Nt+1
        println("n = $n")
        Evo_Slab_Right!(slab_write,slab_read,n)
        @batch for j in 1:Nx+1, i in 1:Nx+1
            W_storage[i,j,n] = slab_read[i,j,3,3]
        end
    end

    h5open("V4_W_fine.h5","w") do io
        write(io,"V4_W_fine.h5",W_storage)
    end

    return
end

Solve_Wightman()
