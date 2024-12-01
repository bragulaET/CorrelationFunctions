using Plots 
using HDF5
using Polyester

const σx = 0.01
const σt = 0.005
const L = 10.0
const dx = 0.01
const dt = dx/2
const Nx = 1000
const Nt = 2000
const iter = 1
const cfl = dt/dx
const cfl2 = cfl^2

const loc_cent = 5.0
const Nmodes = 100
const mode_init = 1
const α = 1/5

const vac_modes = 50
const IC_type = "loc" #"loc" or "vac"

const λ = sqrt(abs(sum((exp(-(((pi^2*α^2)/L^2)*n^2/2)))^2 for n in mode_init:Nmodes)))

Wightman_vac(x::Real,y::Real) = (1/pi)*sum(((1/N)*exp(-(N^2*pi^2*σx^2)/L^2)*exp(-(N^2*pi^2*σt^2)/L^2)*
                        sinpi((N*x)/L)*sinpi((N*y)/L)) for N in 1:vac_modes)  

∂tWightman_vac(x::Real,y::Real) = (-im/L)*sum((exp(-(N^2*pi^2*σx^2)/L^2)*exp(-(N^2*pi^2*σt^2)/L^2)*
                        sinpi((N*x)/L)*sinpi((N*y)/L)) for N in 1:vac_modes)

∂τWightman_vac(x::Real,y::Real) = (im/L)*sum((exp(-(N^2*pi^2*σx^2)/L^2)*exp(-(N^2*pi^2*σt^2)/L^2)*
                        sinpi((N*x)/L)*sinpi((N*y)/L)) for N in 1:vac_modes)

∂t∂τWightman_vac(x::Real,y::Real) = (pi/(L^2))*sum(N*(exp(-(N^2*pi^2*σx^2)/L^2)*exp(-(N^2*pi^2*σt^2)/L^2)*
                        sinpi((N*x)/L)*sinpi((N*y)/L)) for N in 1:vac_modes)

mode_function(x::Real,N::Int64) = sqrt(2/l_sym)*sinpi((N*x)/l_sym)

loc_f(n::Int64) = (1/λ)*exp(-(((pi^2*α^2)/L^2)*n^2)/2)*exp(im*((n*pi*loc_cent)/L))

loc_g(n::Int64) = (1/λ)*exp(-(((pi^2*α^2)/L^2)*n^2)/2)*exp(-im*((n*pi*loc_cent)/L))

Wightman_loc(x::Real,y::Real,N::Int64) = (1/pi)*sum(((loc_g(r)*loc_f(s))/(sqrt(r*s)))*(sin((s*pi*x)/L)*sin((r*pi*y)/L)+sin((r*pi*x)/L)*sin((s*pi*y)/L)) for r in mode_init:N, s in mode_init:N)

∂tWightman_loc(x::Real,y::Real,N::Int64) = (-im/L)*sum(((loc_g(r)*loc_f(s)))*(sqrt(s/r)*sin((s*pi*x)/L)*sin((r*pi*y)/L)-sqrt(r/s)*sin((r*pi*x)/L)*sin((s*pi*y)/L)) for r in mode_init:N, s in mode_init:N)

∂τWightman_loc(x::Real,y::Real,N::Int64) = (-im/L)*sum(((loc_g(r)*loc_f(s)))*(sqrt(r/s)*sin((s*pi*x)/L)*sin((r*pi*y)/L)-sqrt(s/r)*sin((r*pi*x)/L)*sin((s*pi*y)/L)) for r in mode_init:N, s in mode_init:N)

∂t∂τWightman_loc(x::Real,y::Real,N::Int64) = (pi/L^2)*sum(((loc_g(r)*loc_f(s))*(sqrt(r*s)))*(sin((s*pi*x)/L)*sin((r*pi*y)/L)+sin((r*pi*x)/L)*sin((s*pi*y)/L)) for r in mode_init:N, s in mode_init:N)

function IC1!(slab_read::AbstractArray{<:Complex,4},x::Vector{Float64},IC_type::String)
    """Fills in slab_read with the values of the chosen initial condition IC_type
       at every point on the grid x.
       slab_read: AbstractArray, holds the values of the Wightman at the grid points
       x: Vector, all grid points on the computational domain
       IC_type: String, determines whether the IC is obtained from localalized or vacuum wightman
    """
    println("Running IC1")
    if IC_type == "loc"

        @batch for j in 1:Nx+1, i in 1:Nx+1
            slab_read[i,j,1,1] = Wightman_loc(x[i],x[j],Nmodes)
        end

    elseif IC_type == "vac"
        @batch for j in 1:Nx+1, i in 1:Nx+1
            slab_read[i,j,1,1] = Wightman_vac(x[i],x[j])
        end
    else
        println("No Initial Condition $IC_type exists.")
    end
    return
end

function IC2!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{ComplexF64},x::Vector{Float64},IC_type::String)
    """Fills in slab_write by evolving the initial condition forward one time step in the t' direction.
       slab_read: AbstractArray, holds the values of the Wightman at the grid points prior
       slab_write: AbstractArray, holds the values of the Wightman at the first time step in the t' direction
       x: Vector, all grid points on the computational domain
       IC_type: String, determines whether the IC is obtained from localalized or vacuum wightman
    """
    println("Running IC2")
    if IC_type == "loc"
        @batch for j in 2:Nx, i in 2:Nx
            slab_write[i,j,1,2] = slab_read[i,j,1,1] + (1/2)*cfl2*(slab_read[i,j+1,1,1] - 2*slab_read[i,j,1,1] + slab_read[i,j-1,1,1]) + dt*∂τWightman_loc(x[i],x[j],Nmodes)

            slab_write[i,1,1,2] = 0.0
            slab_write[i,end,1,2] = 0.0
            slab_write[1,j,1,2] = 0.0
            slab_write[end,j,1,2] = 0.0
        end

    elseif IC_type == "vac"
        @batch for j in 2:Nx, i in 2:Nx
            slab_write[i,j,1,2] = slab_read[i,j,1,1] + (1/2)*cfl2*(slab_read[i,j+1,1,1] - 2*slab_read[i,j,1,1] + slab_read[i,j-1,1,1]) + dt*∂τWightman_vac(x[i],x[j])
            
            slab_write[i,1,1,2] = 0.0
            slab_write[i,end,1,2] = 0.0
            slab_write[1,j,1,2] = 0.0
            slab_write[end,j,1,2] = 0.0
        end

    else
        println("No Initial Condition $IC_type exists.")
    end
    return 
end

function IC3!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4},x::Vector{Float64},IC_type::String)
    """Fills in slab_write by evolving the initial condition forward one time step in the t direction.
       slab_read: AbstractArray, holds the values of the Wightman at the grid points prior
       slab_write: AbstractArray, holds the values of the Wightman at the first time step in the t' direction
       x: Vector, all grid points on the computational domain
       IC_type: String, determines whether the IC is obtained from localalized or vacuum wightman
    """
    println("Running IC3")
    if IC_type == "loc"
        @batch for j in 2:Nx, i in 2:Nx
            slab_write[i,j,2,1] = slab_read[i,j,1,1] +(1/2)*cfl2*(slab_read[i+1,j,1,1] - 2*slab_read[i,j,1,1] + slab_read[i-1,j,1,1]) + dt*∂tWightman_loc(x[i],x[j],Nmodes)

            slab_write[i,1,2,1] = 0.0
            slab_write[i,end,2,1] = 0.0
            slab_write[1,j,2,1] = 0.0
            slab_write[end,j,2,1] = 0.0
        end

    elseif IC_type == "vac"
        @batch for j in 2:Nx, i in 2:Nx
            slab_write[i,j,2,1] = slab_read[i,j,1,1] +(1/2)*cfl2*(slab_read[i+1,j,1,1] - 2*slab_read[i,j,1,1] + slab_read[i-1,j,1,1]) + dt*∂tWightman_vac(x[i],x[j]) 
            
            slab_write[i,1,2,1] = 0.0
            slab_write[i,end,2,1] = 0.0
            slab_write[1,j,2,1] = 0.0
            slab_write[end,j,2,1] = 0.0
        end

    else
        println("No Initial Condition $IC_type exists.")
    end
    return
end

function IC4!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4},x::Vector{Float64},IC_type::String)
    """Fills in slab_write by evolving the initial condition forward one time step in the t and t' direction.
       slab_read: AbstractArray, holds the values of the Wightman at the grid points prior
       slab_write: AbstractArray, holds the values of the Wightman at the first time step in the t and t' direction
       x: Vector, all grid points on the computational domain
       IC_type: String, determines whether the IC is obtained from localalized or vacuum wightman
    """
    println("Running IC4")
    if IC_type == "loc"
        @batch for j in 2:Nx, i in 2:Nx
            slab_write[i,j,2,2] = slab_read[i,j,1,2] + slab_read[i,j,2,1] - slab_read[i,j,1,1] + dt^2*∂t∂τWightman_loc(x[i],x[j],Nmodes)
    
            slab_write[1,j,2,2] = 0.0
            slab_write[end,j,2,2] = 0.0
            slab_write[i,1,2,2] = 0.0
            slab_write[i,end,2,2] = 0.0
        end

    elseif IC_type == "vac"
        @batch for j in 2:Nx, i in 2:Nx
            slab_write[i,j,2,2] = slab_read[i,j,1,2] + slab_read[i,j,2,1] - slab_read[i,j,1,1] + dt^2*∂t∂τWightman_vac(x[i],x[j])

            slab_write[1,j,2,2] = 0.0
            slab_write[end,j,2,2] = 0.0
            slab_write[i,1,2,2] = 0.0
            slab_write[i,end,2,2] = 0.0
        end

    else
        println("No Initial Condition $IC_type exists.")
    end

    return
end

function Update_IC!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4})
    """ This writes the values in slab_write into the corresponding place in slab_read in order to avoid problems with 
        parallelization later.
        slab_write: AbstractArray, Storage array for the resulting evolution values
        slab_read: AbstractArray, Values from previous evolutions used to solve for the new values of slab_write
    """

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

function Make_Slab!(slab_write::AbstractArray{<:Complex,4}, slab_read::AbstractArray{<:Complex,4},V::AbstractArray{<:Complex,2})
    """ This function fills out the initial 3x3 (t,t')-steps into the slab_write variable. The final step is then 
        to copy this data into slab_read so that we can evolve the slab forward later on.
        slab_write: AbstractArray, Storage array for the resulting evolution values
        slab_read: AbstractArray, Values from previous evolutions used to solve for the new values of slab_write
    """
    println("Making Initial Slab")
    #This first loop evolves forward to the t' = 3dt time step at the first two t-steps.
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,1,3] = 2*slab_read[i,j,1,2] - slab_read[i,j,1,1] + cfl2*(slab_read[i,j+1,1,2] - 2*slab_read[i,j,1,2] + slab_read[i,j-1,1,2]) - 2*dt^2*V[j,2]*slab_read[i,j,1,2]
        slab_write[i,j,2,3] = 2*slab_read[i,j,2,2] - slab_read[i,j,2,1] + cfl2*(slab_read[i,j+1,2,2] - 2*slab_read[i,j,2,2] + slab_read[i,j-1,2,2]) - 2*dt^2*V[j,2]*slab_read[i,j,2,2]
    end

    Apply_BCs!(slab_write,1,3)
    Apply_BCs!(slab_write,2,3)

    Write_to_Read!(slab_write,slab_read,1,3)
    Write_to_Read!(slab_write,slab_read,2,3)

    #This second loop evolves forward to the t = 3dt time step for the first three t'-steps.
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,3,1] = 2*slab_read[i,j,2,1] - slab_read[i,j,1,1] + cfl2*(slab_read[i+1,j,2,1] - 2*slab_read[i,j,2,1] + slab_read[i-1,j,2,1]) - 2*dt^2*V[i,2]*slab_read[i,j,2,1]
        slab_write[i,j,3,2] = 2*slab_read[i,j,2,2] - slab_read[i,j,1,2] + cfl2*(slab_read[i+1,j,2,2] - 2*slab_read[i,j,2,2] + slab_read[i-1,j,2,2]) - 2*dt^2*V[i,2]*slab_read[i,j,2,2]
        slab_write[i,j,3,3] = 2*slab_read[i,j,2,3] - slab_read[i,j,1,3] + cfl2*(slab_read[i+1,j,2,3] - 2*slab_read[i,j,2,3] + slab_read[i-1,j,2,3]) - 2*dt^2*V[i,2]*slab_read[i,j,2,3]
    end

    Apply_BCs!(slab_write,3,1)
    Apply_BCs!(slab_write,3,2)
    Apply_BCs!(slab_write,3,3)

    Write_to_Read!(slab_write,slab_read,3,1)
    Write_to_Read!(slab_write,slab_read,3,2)
    Write_to_Read!(slab_write,slab_read,3,3)  
    
    return 
end

function Evo_Slab_Up!(slab_write::AbstractArray{<:Complex,4}, slab_read::AbstractArray{<:Complex,4}, V::AbstractArray{<:Complex,2}, m::Int64)
    
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,1,3] = 2*slab_read[i,j,1,3] - slab_read[i,j,1,2] + cfl2*(slab_read[i,j+1,1,3] - 2*slab_read[i,j,1,3] + slab_read[i,j-1,1,3]) - 2*dt^2*V[j,m-1]*slab_read[i,j,1,3]
        slab_write[i,j,2,3] = 2*slab_read[i,j,2,3] - slab_read[i,j,2,2] + cfl2*(slab_read[i,j+1,2,3] - 2*slab_read[i,j,2,3] + slab_read[i,j-1,2,3]) - 2*dt^2*V[j,m-1]*slab_read[i,j,2,3]
        slab_write[i,j,3,3] = 2*slab_read[i,j,3,3] - slab_read[i,j,3,2] + cfl2*(slab_read[i,j+1,3,3] - 2*slab_read[i,j,3,3] + slab_read[i,j-1,3,3]) - 2*dt^2*V[j,m-1]*slab_read[i,j,3,3]
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

function Evo_Slab_Right!(slab_write::AbstractArray{<:Complex,4}, slab_read::AbstractArray{<:Complex,4}, V::AbstractArray{<:Complex,2}, m::Int64)
    
    @batch for j in 2:Nx, i in 2:Nx
        slab_write[i,j,3,1] = 2*slab_read[i,j,3,1] - slab_read[i,j,2,1] + cfl2*(slab_read[i+1,j,3,1] - 2*slab_read[i,j,3,1] + slab_read[i-1,j,3,1]) - 2*dt^2*V[i,m-1]*slab_read[i,j,3,1]
        slab_write[i,j,3,2] = 2*slab_read[i,j,3,2] - slab_read[i,j,2,2] + cfl2*(slab_read[i+1,j,3,2] - 2*slab_read[i,j,3,2] + slab_read[i-1,j,3,2]) - 2*dt^2*V[i,m-1]*slab_read[i,j,3,2]
        slab_write[i,j,3,3] = 2*slab_read[i,j,3,3] - slab_read[i,j,2,3] + cfl2*(slab_read[i+1,j,3,3] - 2*slab_read[i,j,3,3] + slab_read[i-1,j,3,3]) - 2*dt^2*V[i,m-1]*slab_read[i,j,3,3]
    end

    Apply_BCs!(slab_write,3,1)
    Apply_BCs!(slab_write,3,2)
    Apply_BCs!(slab_write,3,3)

    for j in 1:Nx, i in 1:Nx
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

function Diagonal_Evolution!(slab_write::AbstractArray{<:Complex,4}, slab_read::AbstractArray{<:Complex,4},V::AbstractArray{<:Complex,2}, m::Int64)

    Evo_Slab_Up!(slab_write,slab_read,V,m)
    Evo_Slab_Right!(slab_write,slab_read,V,m)

    return
end

function calc_T00_x!(T00_x::AbstractArray{<:Complex,2},strip::AbstractArray{<:Complex,2},i::Int64)
    if i==1
        T00_x[i,i] = (1/dx^2)*(strip[2,2]-strip[1,2]-strip[2,1]+strip[1,1])
    elseif i==Nx+1
        T00_x[i,i] = (1/dx^2)*(strip[end,end]-strip[end-1,end]-strip[end,end-1]+strip[end-1,end-1])
    else
        T00_x[i,i] = (1/(2*dx)^2)*(strip[i+1,i+1]-strip[i-1,i+1]-strip[i+1,i-1]+strip[i-1,i-1])
    end
    
    return
end

function calc_T00_Full!(slab_write::AbstractArray{<:Complex,4},slab_read::AbstractArray{<:Complex,4},V::AbstractArray{<:Complex,2},
                        T00_full::AbstractArray{<:Complex,2},T00_t::AbstractArray{<:Complex,2},T00_x::AbstractArray{<:Complex,2},strip_x::AbstractArray{<:Complex,2})

    for m in 1:Nt+1
        println("m = $m")
        if mod(m-1,iter) == 0
            if m == 1
                @batch for j in 1:Nx+1, i in 1:Nx+1
                    T00_t[i,j] = (1/dt^2)*(slab_read[i,j,2,2] - slab_read[i,j,1,2]-slab_read[i,j,2,1]+slab_read[i,j,1,1])
                    strip_x[i,j] = slab_read[i,j,1,1]
                end

                for i in 1:Nx+1
                    calc_T00_x!(T00_x,strip_x,i)
                    T00_full[i,Int((m+iter-1)/iter)] = (1/2)*(T00_t[i,i] + T00_x[i,i]) + V[i,m]*strip_x[i,i]
                end

            elseif m == 2
                @batch for j in 1:Nx+1, i in 1:Nx+1
                    T00_t[i,j] = (1/(2*dt)^2)*(slab_read[i,j,3,3]-slab_read[i,j,1,3]-slab_read[i,j,3,1]+slab_read[i,j,1,1])
                    strip_x[i,j] = slab_read[i,j,2,2]
                end

                for i in 1:Nx+1
                    calc_T00_x!(T00_x,strip_x,i)
                    T00_full[i,Int((m+iter-1)/iter)] = (1/2)*(T00_t[i,i]+T00_x[i,i]) + V[i,m]*strip_x[i,i]
                end

            else
                Diagonal_Evolution!(slab_write,slab_read,V,m)

                @batch for j in 1:Nx+1, i in 1:Nx+1
                    T00_t[i,j] = (1/(2*dt)^2)*(slab_read[i,j,3,3]-slab_read[i,j,1,3]-slab_read[i,j,3,1]+slab_read[i,j,1,1])
                    strip_x[i,j] = slab_read[i,j,2,2]
                end

                for i in 1:Nx+1
                    calc_T00_x!(T00_x,strip_x,i)
                    T00_full[i,Int((m+iter-1)/iter)] = (1/2)*(T00_t[i,i]+T00_x[i,i]) + V[i,m]*strip_x[i,i]
                end

            end

        end

    end

    return
end

function main_T00()
    slab_read = Array{ComplexF64}(undef,Nx+1,Nx+1,3,3)
    slab_write = Array{ComplexF64}(undef,Nx+1,Nx+1,3,3)
    V = Array{ComplexF64}(undef,Nx+1,Nt+1) 
    
    x = [round(0+i*dx,sigdigits = 6) for i in 0:Nx]
    t = [round(0+i*dt,sigdigits = 6) for i in 0:Nt]

    T00_x = Array{ComplexF64}(undef,Nx+1,Nx+1)
    T00_t = Array{ComplexF64}(undef,Nx+1,Nx+1)
    T00_full = Array{ComplexF64}(undef,Nx+1,Int(Nt/iter)+1)
    strip_x = Array{ComplexF64}(undef,Nx+1,Nx+1)

    Evaluate_Potential!(V,x,t,status,β,l,x_l,x_r,V_max,Height,t_stop)
    IC1!(slab_read,x,IC_type)
    IC2!(slab_write,slab_read,x,IC_type)
    IC3!(slab_write,slab_read,x,IC_type)
    Update_IC!(slab_write,slab_read)
    IC4!(slab_write,slab_read,x,IC_type)
    Update_IC!(slab_write,slab_read)
    Make_Slab!(slab_write,slab_read,V)

    calc_T00_Full!(slab_write,slab_read,V,T00_full,T00_t,T00_x,strip_x)

    h5open("EvTT00.h5","w") do io 
        write(io,"EvTT00",T00_full)
    end
    return
end

main_T00()