#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : types.jl                                                                   |
|  Description: Type system in MPMSolver.jl                                                |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Struct     : 1. Args2D/3D                                                               |
|               2. Grid2D/3D                                                               |
|               3. Particle2D/3D                                                           |
|               4. VBoundary2D/3D                                                          |
+==========================================================================================#

abstract type      ARGS end
abstract type      GRID end
abstract type  PARTICLE end
abstract type       VBC end

#=-----------------------------------------------------------------------------------------+
| 1. Struct of Args2D/3D and terminal outputs                                              |
+-----------------------------------------------------------------------------------------=#
@kwdef mutable struct Args2D{T1<:Signed, T2<:AbstractFloat} <: ARGS
    const Ttol        ::T2
    const Te          ::T2
          ΔT          ::T2
    const time_step   ::Symbol  = :auto
    const FLIP        ::T2
    const PIC         ::T2
    const constitutive::Symbol
    const basis       ::Symbol
    const animation   ::Bool    = false
    const hdf5        ::Bool    = false
    const hdf5_step   ::T1
    const vollock     ::Bool    = false
    const device      ::Symbol  = :CPU
    const coupling    ::Symbol  = :OS
    const progressbar ::Bool    = true
    const gravity     ::T2      = T2(-9.8)
    const ζ           ::T2      = T2(0)
    const αT          ::T2      = T2(0.5)
          iter_num    ::Int64   = Int64(0)
          end_time    ::Float64 = Float64(0)
          start_time  ::Float64 = Float64(0)
    const project_name::String
    const project_path::String
    function Args2D{T1, T2}(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, basis, 
        animation, hdf5, hdf5_step, vollock, device, coupling, progressbar, gravity, ζ, αT, 
        iter_num, end_time, start_time, project_name, project_path) where {T1, T2}
        # project default value
        rm(project_path, recursive=true, force=true); mkpath(project_path)
        con_set = [:linearelastic, :hyperelastic, :druckerprager, :mohrcoulomb, :taitwater]
        cop_set = [:OS, :TS]
        bas_set = [:uGIMP, :linear]
        dev_set = [:CPU, :CUDA]
        tis_set = [:fixed, :auto]
        # parameter check
        0<Ttol                  ? nothing : error("Simulation time cannot be ≤0 s."        )
        ΔT≤Ttol                 ? nothing : error("Time step cannot be >$(Tol)s."          )
        0≤Te≤Ttol               ? nothing : error("Elastic loading time must in 0~$(Tol)s.")
        basis in bas_set        ? nothing : error("Cannot find $(basis) basis function."   )
        (FLIP+PIC)==1           ? nothing : error("FLIP+PIC must be 1."                    )
        0≤αT≤1                  ? nothing : error("Time step factor error."                )
        project_name≠""         ? nothing : error("Empty project name."                    )
        ispath(project_path)    ? nothing : error("Project path error."                    )
        constitutive in con_set ? nothing : error("Constitutive model's name error."       )
        coupling in cop_set     ? nothing : error("Coupling mode is wrong."                )
        device in dev_set       ? nothing : error("Cannot find $(device) device."          )
        time_step in tis_set    ? nothing : error("$(time_step) time step is not allowed." )
        (animation==true)&&(hdf5==false) ? (hdf5=true; @warn "HDF5 forced ON due to the animation") : nothing
        (hdf5==true)&&(hdf5_step≤0)  ? error("HDF5 step cannot be ≤0.") : nothing
        project_path = abspath(project_path)
        # update struct
        new(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, basis, animation, hdf5, 
            hdf5_step, vollock, device, coupling, progressbar, gravity, ζ, αT, iter_num, 
            end_time, start_time, project_name, project_path)
    end
end

@kwdef mutable struct Args3D{T1<:Signed, T2<:AbstractFloat} <: ARGS
    const Ttol        ::T2
    const Te          ::T2
          ΔT          ::T2
    const time_step   ::Symbol  = :auto
    const FLIP        ::T2
    const PIC         ::T2
    const constitutive::Symbol
    const basis       ::Symbol
    const animation   ::Bool    = false
    const hdf5        ::Bool    = false
    const hdf5_step   ::T1
    const vollock     ::Bool    = false
    const device      ::Symbol  = :CPU
    const coupling    ::Symbol  = :OS
    const progressbar ::Bool    = true
    const gravity     ::T2      = T2(-9.8)
    const ζ           ::T2      = T2(0)
    const αT          ::T2      = T2(0.5)
          iter_num    ::Int64   = Int64(0)
          end_time    ::Float64 = Float64(0)
          start_time  ::Float64 = Float64(0)
    const project_name::String
    const project_path::String
    function Args3D{T1, T2}(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, basis, 
        animation, hdf5, hdf5_step, vollock, device, coupling, progressbar, gravity, ζ, αT, 
        iter_num, end_time, start_time, project_name, project_path) where {T1, T2}
        # project default value
        rm(project_path, recursive=true, force=true); mkpath(project_path)
        con_set = [:linearelastic, :hyperelastic, :druckerprager, :mohrcoulomb, :taitwater]
        cop_set = [:OS, :TS]
        bas_set = [:uGIMP, :linear]
        dev_set = [:CPU, :CUDA]
        tis_set = [:fixed, :auto]
        # parameter check
        0<Ttol                  ? nothing : error("Simulation time cannot be ≤0 s."        )
        ΔT≤Ttol                 ? nothing : error("Time step cannot be >$(Tol)s."          )
        0≤Te≤Ttol               ? nothing : error("Elastic loading time must in 0~$(Tol)s.")
        basis in bas_set        ? nothing : error("Cannot find $(basis) basis function."   )
        (FLIP+PIC)==1           ? nothing : error("FLIP+PIC must be 1."                    )
        0≤αT≤1                  ? nothing : error("Time step factor error."                )
        project_name≠""         ? nothing : error("Empty project name."                    )
        ispath(project_path)    ? nothing : error("Project path error."                    )
        constitutive in con_set ? nothing : error("Constitutive model's name error."       )
        coupling in cop_set     ? nothing : error("Coupling mode is wrong."                )
        device in dev_set       ? nothing : error("Cannot find $(device) device."          )
        time_step in tis_set    ? nothing : error("$(time_step) time step is not allowed." )
        (animation==true)&&(hdf5==false) ? (hdf5=true; @warn "HDF5 forced ON due to the animation") : nothing
        (hdf5==true)&&(hdf5_step≤0)  ? error("HDF5 step cannot be ≤0.") : nothing
        project_path = abspath(project_path)
        # update struct
        new(Ttol, Te, ΔT, time_step, FLIP, PIC, constitutive, basis, animation, hdf5, 
            hdf5_step, vollock, device, coupling, progressbar, gravity, ζ, αT, iter_num, 
            end_time, start_time, project_name, project_path)
    end
end

function Base.show(io::IO, args::ARGS)
    typeof(args).parameters[2]==Float64 ? precision="FP64" : 
    typeof(args).parameters[2]==Float32 ? precision="FP32" : nothing
    # printer
    print(io, typeof(args)                       , "\n")
    print(io, "─"^length(string(typeof(args)))   , "\n")
    print(io, "project name: ", args.project_name, "\n")
    print(io, "project path: ", args.project_path, "\n")
    print(io, "precision   : ", precision        , "\n")
    print(io, "constitutive: ", args.constitutive, "\n")
    print(io, "basis method: ", args.basis       , "\n")
    print(io, "elim vollock: ", args.vollock     , "\n")
    print(io, "coupling    : ", args.coupling    , "\n")
    return nothing
end

#=-----------------------------------------------------------------------------------------+
| 2. Struct of Grid2D/3D and terminal outputs                                              |
+-----------------------------------------------------------------------------------------=#
@kwdef mutable struct Grid2D{T1<:Signed, T2<:AbstractFloat} <: GRID
    const range_x1  ::T2
    const range_x2  ::T2
    const range_y1  ::T2
    const range_y2  ::T2
    const space_x   ::T2
    const space_y   ::T2
    const phase     ::T1
    const node_num_x::T1 = 0
    const node_num_y::T1 = 0
    const node_num  ::T1 = 0
    const NIC       ::T1 = 16
    const pos       ::Array{T2, 2} = [0 0]
    const cell_num_x::T1 = 0
    const cell_num_y::T1 = 0
    const cell_num  ::T1 = 0
    const c2n       ::Array{T1, 2} = [0 0]
          σm        ::Array{T2, 1} = [0]
          σw        ::Array{T2, 1} = [0]
          vol       ::Array{T2, 1} = [0]
          Ms        ::Array{T2, 1} = [0]
          Mw        ::Array{T2, 1} = [0]
          Mi        ::Array{T2, 1} = [0]
          Ps        ::Array{T2, 2} = [0 0]
          Pw        ::Array{T2, 2} = [0 0]
          Vs        ::Array{T2, 2} = [0 0]
          Vw        ::Array{T2, 2} = [0 0]
          Vs_T      ::Array{T2, 2} = [0 0]
          Vw_T      ::Array{T2, 2} = [0 0]
          Fs        ::Array{T2, 2} = [0 0]
          Fw        ::Array{T2, 2} = [0 0]
          Fdrag     ::Array{T2, 2} = [0 0]
          a_s       ::Array{T2, 2} = [0 0]
          a_w       ::Array{T2, 2} = [0 0]
          Δd_s      ::Array{T2, 2} = [0 0]
          Δd_w      ::Array{T2, 2} = [0 0]
    function Grid2D{T1, T2}(range_x1, range_x2, range_y1, range_y2, space_x, space_y, phase,
        node_num_x, node_num_y, node_num, NIC, pos, cell_num_x, cell_num_y, cell_num, c2n, 
        σm, σw, vol, Ms, Mw, Mi, Ps, Pw, Vs, Vw, Vs_T, Vw_T, Fs, Fw, Fdrag, a_s, a_w, Δd_s, 
        Δd_w) where {T1, T2}
        # necessary input check
        DoF = 2
        NIC==4||NIC==16 ? nothing : error("Nodes in Cell can only be 4 or 16.")
        # set the nodes in background grid
        vx = range_x1:space_x:range_x2 |> collect
        vy = range_y1:space_y:range_y2 |> collect
        sort!(vx); sort!(vy, rev=true) # vy should from largest to smallest
        node_num_x = length(vx); vx = reshape(vx, 1, node_num_x)
        node_num_y = length(vy); vy = reshape(vy, node_num_y, 1)
        node_num   = node_num_y*node_num_x
        x          = repeat(vx, node_num_y, 1) |> vec
        y          = repeat(vy, 1, node_num_x) |> vec
        pos        = hcat(x, y)
        # set the cells in background grid
        cell_num_x = node_num_x-1
        cell_num_y = node_num_y-1
        cell_num   = cell_num_y*cell_num_x
        # grid properties setup
        phase==1 ? (cell_new=1       ; node_new=1       ; DoF_new=1  ) : 
        phase==2 ? (cell_new=cell_num; node_new=node_num; DoF_new=DoF) : nothing
        σm       = Array{T2, 1}(calloc, cell_num         )
        σw       = Array{T2, 1}(calloc, cell_new         )
        vol      = Array{T2, 1}(calloc, cell_num         )
        Ms       = Array{T2, 1}(calloc, node_num         )
        Mw       = Array{T2, 1}(calloc, node_new         )
        Mi       = Array{T2, 1}(calloc, node_new         )
        c2n      = Array{T1, 2}(calloc, cell_num, NIC    )
        Ps       = Array{T2, 2}(calloc, node_num, DoF    )
        Pw       = Array{T2, 2}(calloc, node_new, DoF_new)
        Vs       = Array{T2, 2}(calloc, node_num, DoF    )
        Vw       = Array{T2, 2}(calloc, node_new, DoF_new)
        Vs_T     = Array{T2, 2}(calloc, node_num, DoF    )
        Vw_T     = Array{T2, 2}(calloc, node_new, DoF_new)
        a_s      = Array{T2, 2}(calloc, node_num, DoF    )
        a_w      = Array{T2, 2}(calloc, node_new, DoF_new)
        Fs       = Array{T2, 2}(calloc, node_num, DoF    )
        Fw       = Array{T2, 2}(calloc, node_new, DoF_new)
        Fdrag    = Array{T2, 2}(calloc, node_new, DoF_new)
        Δd_s     = Array{T2, 2}(calloc, node_num, DoF    )
        Δd_w     = Array{T2, 2}(calloc, node_new, DoF_new)
        # set the computing cell to node topology
        if NIC == 4
            for i in 1:cell_num
                col_id    = cld(i, cell_num_y)       # belongs to which column
                row_id    = i-(col_id-1)*cell_num_y  # belongs to which row
                ltn_id    = col_id*node_num_y-row_id # top    left  node's index              
                c2n[i, 1] = ltn_id+1                 # bottom left  node
                c2n[i, 2] = ltn_id+0                 # top    left  node
                c2n[i, 3] = ltn_id+node_num_y+1      # bottom right node
                c2n[i, 4] = ltn_id+node_num_y        # top    right node
            end
        elseif NIC == 16
            for i in 1:cell_num
                col_id    = cld(i, cell_num_y)       # belongs to which column
                row_id    = i-(col_id-1)*cell_num_y  # belongs to which row
                ltn_id    = col_id*node_num_y-row_id # top left  node's index    
                if (1<col_id<cell_num_x)&&(1<row_id<cell_num_y)
                    c2n[i,  1] = ltn_id-node_num_y+2                
                    c2n[i,  2] = ltn_id-node_num_y+1                
                    c2n[i,  3] = ltn_id-node_num_y+0   
                    c2n[i,  4] = ltn_id-node_num_y-1       
                    c2n[i,  5] = ltn_id+2                 
                    c2n[i,  6] = ltn_id+1              
                    c2n[i,  7] = ltn_id+0     
                    c2n[i,  8] = ltn_id-1     
                    c2n[i,  9] = ltn_id+node_num_y+2                
                    c2n[i, 10] = ltn_id+node_num_y+1                
                    c2n[i, 11] = ltn_id+node_num_y+0      
                    c2n[i, 12] = ltn_id+node_num_y-1      
                    c2n[i, 13] = ltn_id+node_num_y*2+2                 
                    c2n[i, 14] = ltn_id+node_num_y*2+1                 
                    c2n[i, 15] = ltn_id+node_num_y*2+0      
                    c2n[i, 16] = ltn_id+node_num_y*2-1        
                end
            end
        end
        # update struct
        new(range_x1, range_x2, range_y1, range_y2, space_x, space_y, phase, node_num_x, 
            node_num_y, node_num, NIC, pos, cell_num_x, cell_num_y, cell_num, c2n, σm, σw, 
            vol, Ms, Mw, Mi, Ps, Pw, Vs, Vw, Vs_T, Vw_T, Fs, Fw, Fdrag, a_s, a_w, Δd_s, 
            Δd_w)
    end
end

@kwdef mutable struct Grid3D{T1<:Signed, T2<:AbstractFloat} <: GRID
    const range_x1  ::T2
    const range_x2  ::T2
    const range_y1  ::T2
    const range_y2  ::T2
    const range_z1  ::T2
    const range_z2  ::T2
    const space_x   ::T2
    const space_y   ::T2
    const space_z   ::T2
    const phase     ::T1
    const node_num_x::T1 = 0
    const node_num_y::T1 = 0
    const node_num_z::T1 = 0
    const node_num  ::T1 = 0
    const NIC       ::T1 = 64
    const pos       ::Array{T2, 2} = [0 0]
    const cell_num_x::T1 = 0
    const cell_num_y::T1 = 0
    const cell_num_z::T1 = 0
    const cell_num  ::T1 = 0
    const c2n       ::Array{T1, 2} = [0 0]
          σm        ::Array{T2, 1} = [0]
          σw        ::Array{T2, 1} = [0]
          vol       ::Array{T2, 1} = [0]
          Ms        ::Array{T2, 1} = [0]
          Mw        ::Array{T2, 1} = [0]
          Mi        ::Array{T2, 1} = [0]
          Ps        ::Array{T2, 2} = [0 0]
          Pw        ::Array{T2, 2} = [0 0]
          Vs        ::Array{T2, 2} = [0 0]
          Vw        ::Array{T2, 2} = [0 0]
          Vs_T      ::Array{T2, 2} = [0 0]
          Vw_T      ::Array{T2, 2} = [0 0]
          Fs        ::Array{T2, 2} = [0 0]
          Fw        ::Array{T2, 2} = [0 0]
          Fdrag     ::Array{T2, 2} = [0 0]
          a_s       ::Array{T2, 2} = [0 0]
          a_w       ::Array{T2, 2} = [0 0]
          Δd_s      ::Array{T2, 2} = [0 0]
          Δd_w      ::Array{T2, 2} = [0 0]
    function Grid3D{T1, T2}(range_x1, range_x2, range_y1, range_y2, range_z1, range_z2, 
        space_x, space_y, space_z, phase, node_num_x, node_num_y, node_num_z, node_num, NIC, 
        pos, cell_num_x, cell_num_y, cell_num_z, cell_num, c2n, σm, σw, vol, Ms, Mw, Mi, Ps,
        Pw, Vs, Vw, Vs_T, Vw_T, Fs, Fw, Fdrag, a_s, a_w, Δd_s, Δd_w) where {T1, T2}
        # necessary input check
        DoF = 3
        NIC==8||NIC==64 ? nothing : error("Nodes in Cell can only be 8 or 64.")
        # set the nodes in background grid
        vx   = range_x1:space_x:range_x2 |> collect
        vy   = range_y1:space_y:range_y2 |> collect
        vz   = range_z1:space_z:range_z2 |> collect
        m, n, o = length(vy), length(vx), length(vz)
        vx   = reshape(vx, 1, n, 1)
        vy   = reshape(vy, m, 1, 1)
        vz   = reshape(vz, 1, 1, o)
        om   = ones(Int, m)
        on   = ones(Int, n)
        oo   = ones(Int, o)
        x    = vec(vx[om, :, oo])
        y    = vec(vy[:, on, oo])
        z    = vec(vz[om, on, :])
        pos  = hcat(x, y, z)
        node_num_x = length(vx)
        node_num_y = length(vy)
        node_num_z = length(vz)
        node_num   = node_num_x*node_num_y*node_num_z
        # set the cells in background grid
        cell_num_x = node_num_x-1
        cell_num_y = node_num_y-1
        cell_num_z = node_num_z-1
        cell_num   = cell_num_x*cell_num_y*cell_num_z
        # grid properties setup
        phase==1 ? (cell_new=1       ; node_new=1       ; DoF_new=1  ) : 
        phase==2 ? (cell_new=cell_num; node_new=node_num; DoF_new=DoF) : nothing
        σm      = Array{T2, 1}(calloc, cell_num         )
        σw      = Array{T2, 1}(calloc, cell_new         )
        vol     = Array{T2, 1}(calloc, cell_num         )
        Ms      = Array{T2, 1}(calloc, node_num         )
        Mw      = Array{T2, 1}(calloc, node_new         )
        Mi      = Array{T2, 1}(calloc, node_new         )
        c2n     = Array{T1, 2}(calloc, cell_num, NIC    )
        Ps      = Array{T2, 2}(calloc, node_num, DoF    )
        Pw      = Array{T2, 2}(calloc, node_new, DoF_new)
        Vs      = Array{T2, 2}(calloc, node_num, DoF    )
        Vw      = Array{T2, 2}(calloc, node_new, DoF_new)
        Vs_T    = Array{T2, 2}(calloc, node_num, DoF    )
        Vw_T    = Array{T2, 2}(calloc, node_new, DoF_new)
        a_s     = Array{T2, 2}(calloc, node_num, DoF    )
        a_w     = Array{T2, 2}(calloc, node_new, DoF_new)
        Fs      = Array{T2, 2}(calloc, node_num, DoF    )
        Fw      = Array{T2, 2}(calloc, node_new, DoF_new)
        Fdrag   = Array{T2, 2}(calloc, node_new, DoF_new)
        Δd_s    = Array{T2, 2}(calloc, node_num, DoF    )
        Δd_w    = Array{T2, 2}(calloc, node_new, DoF_new)
        # set the computing cell to node topology
        if NIC==8
            for i in 1:cell_num
                # 处于第几层
                layer = cld(i, cell_num_x*cell_num_y)
                # 处于第几行
                l_idx = cld(i-(layer-1)*cell_num_x*cell_num_y, cell_num_y)
                # 处于第几列
                l_idy = (i-(layer-1)*cell_num_x*cell_num_y)-(l_idx-1)*cell_num_y

                c2n[i, 1] = (layer-1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+1
                c2n[i, 2] = (layer-1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+1
                c2n[i, 3] = (layer-0)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+1
                c2n[i, 4] = (layer-0)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+1
                c2n[i, 5] = (layer-1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+1
                c2n[i, 6] = (layer-1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+1
                c2n[i, 7] = (layer-0)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+1
                c2n[i, 8] = (layer-0)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+1
            end
        elseif NIC==64
            for i in 1:cell_num
                # 处于第几层
                layer = cld(i, cell_num_x*cell_num_y)
                # 处于第几行
                l_idx = cld(i-(layer-1)*cell_num_x*cell_num_y, cell_num_y)
                # 处于第几列
                l_idy = (i-(layer-1)*cell_num_x*cell_num_y)-(l_idx-1)*cell_num_y
                if (1<layer<cell_num_z)&&(1<l_idx<cell_num_x)&&(1<l_idy<cell_num_y)
                    c2n[i,  1] = (layer-1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+0
                    c2n[i,  2] = (layer-1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+0
                    c2n[i,  3] = (layer-0)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+0
                    c2n[i,  4] = (layer-0)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+0
                    c2n[i,  5] = (layer-0)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-1)+0
                    c2n[i,  6] = (layer-1)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-1)+0
                    c2n[i,  7] = (layer-2)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-1)+0
                    c2n[i,  8] = (layer-2)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+0
                    c2n[i,  9] = (layer-2)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+0
                    c2n[i, 10] = (layer-2)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-1)+0
                    c2n[i, 11] = (layer-1)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-1)+0
                    c2n[i, 12] = (layer-0)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-1)+0
                    c2n[i, 13] = (layer+1)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-1)+0
                    c2n[i, 14] = (layer+1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+0
                    c2n[i, 15] = (layer+1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+0
                    c2n[i, 16] = (layer+1)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-1)+0

                    c2n[i, 17] = (layer-1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+1
                    c2n[i, 18] = (layer-1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+1
                    c2n[i, 19] = (layer-0)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+1
                    c2n[i, 20] = (layer-0)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+1
                    c2n[i, 21] = (layer-0)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-1)+1
                    c2n[i, 22] = (layer-1)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-1)+1
                    c2n[i, 23] = (layer-2)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-1)+1
                    c2n[i, 24] = (layer-2)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+1
                    c2n[i, 25] = (layer-2)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+1
                    c2n[i, 26] = (layer-2)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-1)+1
                    c2n[i, 27] = (layer-1)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-1)+1
                    c2n[i, 28] = (layer-0)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-1)+1
                    c2n[i, 29] = (layer+1)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-1)+1
                    c2n[i, 30] = (layer+1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+1
                    c2n[i, 31] = (layer+1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+1
                    c2n[i, 32] = (layer+1)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-1)+1

                    c2n[i, 33] = (layer-1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+1
                    c2n[i, 34] = (layer-1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+1
                    c2n[i, 35] = (layer-0)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+1
                    c2n[i, 36] = (layer-0)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+1
                    c2n[i, 37] = (layer-0)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-0)+1
                    c2n[i, 38] = (layer-1)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-0)+1
                    c2n[i, 39] = (layer-2)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-0)+1
                    c2n[i, 40] = (layer-2)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+1
                    c2n[i, 41] = (layer-2)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+1
                    c2n[i, 42] = (layer-2)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-0)+1
                    c2n[i, 43] = (layer-1)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-0)+1
                    c2n[i, 44] = (layer-0)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-0)+1
                    c2n[i, 45] = (layer+1)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-0)+1
                    c2n[i, 46] = (layer+1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+1
                    c2n[i, 47] = (layer+1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+1
                    c2n[i, 48] = (layer+1)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-0)+1

                    c2n[i, 49] = (layer-1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+2
                    c2n[i, 50] = (layer-1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+2
                    c2n[i, 51] = (layer-0)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+2
                    c2n[i, 52] = (layer-0)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+2
                    c2n[i, 53] = (layer-0)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-0)+2
                    c2n[i, 54] = (layer-1)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-0)+2
                    c2n[i, 55] = (layer-2)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-0)+2
                    c2n[i, 56] = (layer-2)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+2
                    c2n[i, 57] = (layer-2)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+2
                    c2n[i, 58] = (layer-2)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-0)+2
                    c2n[i, 59] = (layer-1)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-0)+2
                    c2n[i, 60] = (layer-0)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-0)+2
                    c2n[i, 61] = (layer+1)*node_num_x*node_num_y+(l_idx+1)*node_num_y+(l_idy-0)+2
                    c2n[i, 62] = (layer+1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+2
                    c2n[i, 63] = (layer+1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+2
                    c2n[i, 64] = (layer+1)*node_num_x*node_num_y+(l_idx-2)*node_num_y+(l_idy-0)+2
            end; end
        end
        # update struct
        new(range_x1, range_x2, range_y1, range_y2, range_z1, range_z2, space_x, space_y, 
            space_z, phase, node_num_x, node_num_y, node_num_z, node_num, NIC, pos, 
            cell_num_x, cell_num_y, cell_num_z, cell_num, c2n, σm, σw, vol, Ms, Mw, Mi, Ps,
            Pw, Vs, Vw, Vs_T, Vw_T, Fs, Fw, Fdrag, a_s, a_w, Δd_s, Δd_w)
    end
end

function Base.show(io::IO, grid::GRID)
    print(io, typeof(grid)                    , "\n")
    print(io, "─"^length(string(typeof(grid))), "\n")
    print(io, "node: ", grid.node_num         , "\n")
    print(io, "cell: ", grid.cell_num         , "\n")
end

#=-----------------------------------------------------------------------------------------+
| 3. Struct of Particle2D/3D and terminal outputs                                          |
+-----------------------------------------------------------------------------------------=#
@kwdef mutable struct Particle2D{T1<:Signed, T2<:AbstractFloat} <: PARTICLE
    const num     ::T1 = 0
    const phase   ::T1
    const NIC     ::T1 = 16
    const space_x ::T2
    const space_y ::T2
          p2c     ::Array{T1, 1} = [0]
          p2n     ::Array{T1, 2} = [0 0]
          pos     ::Array{T2, 2}
          σm      ::Array{T2, 1} = [0]
          J       ::Array{T2, 1} = [0]
          epII    ::Array{T2, 1} = [0]
          epK     ::Array{T2, 1} = [0]
          vol     ::Array{T2, 1} = [0]
    const vol_init::Array{T2, 1} = [0]
          Ms      ::Array{T2, 1} = [0]
          Mw      ::Array{T2, 1} = [0]
          Mi      ::Array{T2, 1} = [0]
          porosity::Array{T2, 1} = [0]
          cfl     ::Array{T2, 1} = [0]
    const k       ::Array{T2, 1} = [0]
          ρs      ::Array{T2, 1}
    const ρs_init ::Array{T2, 1} = [0]
          ρw      ::Array{T2, 1} = [0]
    const ρw_init ::Array{T2, 1} = [0]
    const ν       ::Array{T2, 1}
    const E       ::Array{T2, 1}
    const G       ::Array{T2, 1}
    const Ks      ::Array{T2, 1}
    const Kw      ::Array{T2, 1} = [0]
    const σt      ::Array{T2, 1} = [0] # tensile strength
    const ϕ       ::Array{T2, 1} = [0] # friction angle ϕ
    const c       ::Array{T2, 1} = [0] # cohesion
    const cr      ::Array{T2, 1} = [0] # residual cohesion
    const Hp      ::Array{T2, 1} = [0] # softening modulus
    const ψ       ::Array{T2, 1} = [0] # dilation angle ψ      
    const init    ::Array{T2, 2} = [0 0]
    const γ       ::Array{T2, 1} = [0]
    const B       ::Array{T2, 1} = [0]
          σw      ::Array{T2, 1} = [0]
          σij     ::Array{T2, 2} = [0 0]
          ϵij_s   ::Array{T2, 2} = [0 0]
          ϵij_w   ::Array{T2, 2} = [0 0]
          Δϵij_s  ::Array{T2, 2} = [0 0]
          Δϵij_w  ::Array{T2, 2} = [0 0]
          sij     ::Array{T2, 2} = [0 0]
          Vs      ::Array{T2, 2} = [0 0]
          Vw      ::Array{T2, 2} = [0 0]
          Ps      ::Array{T2, 2} = [0 0]
          Pw      ::Array{T2, 2} = [0 0]
          Ni      ::Array{T2, 2} = [0 0]
          ∂Nx     ::Array{T2, 2} = [0 0]
          ∂Ny     ::Array{T2, 2} = [0 0]
          ∂Fs     ::Array{T2, 2} = [0 0]
          ∂Fw     ::Array{T2, 2} = [0 0]
          F       ::Array{T2, 2} = [0 0]
    const layer   ::Array{T1, 1}
    function Particle2D{T1, T2}(num, phase, NIC, space_x, space_y, p2c, p2n, pos, σm, J, 
        epII, epK, vol, vol_init, Ms, Mw, Mi, porosity, cfl, k, ρs, ρs_init, ρw, ρw_init, ν, 
        E, G, Ks, Kw, σt, ϕ, c, cr, Hp, ψ, init, γ, B, σw, σij, ϵij_s, ϵij_w, Δϵij_s, 
        Δϵij_w, sij, Vs, Vw, Ps, Pw, Ni, ∂Nx, ∂Ny, ∂Fs, ∂Fw, F, layer) where {T1, T2}
        # necessary input check
        DoF = 2
        NIC==4||NIC==16       ? nothing : error("Nodes in Cell can only be 4 or 16.")
        space_x!=zeros(T2, 1) ? nothing : error("Particle x_space is 0."            )
        space_y!=zeros(T2, 1) ? nothing : error("Particle y_space is 0."            )
        # particles properties setup
        num      = size(pos, 1)
        J        = ones(T2, num)
        vol_init = repeat([space_x*space_y], num)
        ρs_init  = copy(ρs)
        ρw_init  = copy(ρw)
        F        = repeat(T2[1 0 0 1] , num)
        vol      = copy(vol_init)
        init     = copy(pos)
        phase==1 ? (num_new=1  ; DoF_new=1  ) : 
        phase==2 ? (num_new=num; DoF_new=DoF) : nothing
        p2c      = Array{T1, 1}(calloc, num             )
        Ms       = Array{T1, 1}(calloc, num             )
        Mw       = Array{T1, 1}(calloc, num_new         )
        Mi       = Array{T1, 1}(calloc, num_new         )
        epII     = Array{T2, 1}(calloc, num             )
        epK      = Array{T2, 1}(calloc, num             )
        σw       = Array{T2, 1}(calloc, num_new         )
        σm       = Array{T2, 1}(calloc, num             )
        cfl      = Array{T2, 1}(calloc, num             )
        ϵij_s    = Array{T2, 2}(calloc, num    , 4      )
        ϵij_w    = Array{T2, 2}(calloc, num_new, 4      )
        σij      = Array{T2, 2}(calloc, num    , 4      )
        Δϵij_s   = Array{T2, 2}(calloc, num    , 4      )
        Δϵij_w   = Array{T2, 2}(calloc, num_new, 4      )
        sij      = Array{T2, 2}(calloc, num    , 4      )
        ∂Fs      = Array{T2, 2}(calloc, num    , 4      )
        ∂Fw      = Array{T2, 2}(calloc, num_new, 4      )
        p2n      = Array{T1, 2}(calloc, num    , NIC    )
        Ps       = Array{T2, 2}(calloc, num    , DoF    )
        Pw       = Array{T2, 2}(calloc, num_new, DoF_new)
        Vs       = Array{T2, 2}(calloc, num    , DoF    )
        Vw       = Array{T2, 2}(calloc, num_new, DoF_new)
        Ni       = Array{T2, 2}(calloc, num    , NIC    )
        ∂Nx      = Array{T2, 2}(calloc, num    , NIC    )
        ∂Ny      = Array{T2, 2}(calloc, num    , NIC    )
        # update struct
        new(num, phase, NIC, space_x, space_y, p2c, p2n, pos, σm, J, epII, epK, vol, 
            vol_init, Ms, Mw, Mi, porosity, cfl, k, ρs, ρs_init, ρw, ρw_init, ν, E, G, Ks, 
            Kw, σt, ϕ, c, cr, Hp, ψ, init, γ, B, σw, σij, ϵij_s, ϵij_w, Δϵij_s, Δϵij_w, sij,
            Vs, Vw, Ps, Pw, Ni, ∂Nx, ∂Ny, ∂Fs, ∂Fw, F, layer)
    end
end

@kwdef mutable struct Particle3D{T1<:Signed, T2<:AbstractFloat} <: PARTICLE
    const num     ::T1 = 0
    const phase   ::T1
    const NIC     ::T1 = 64
    const space_x ::T2
    const space_y ::T2
    const space_z ::T2
          p2c     ::Array{T1, 1} = [0]
          p2n     ::Array{T1, 2} = [0 0]
          pos     ::Array{T2, 2}
          σm      ::Array{T2, 1} = [0]
          J       ::Array{T2, 1} = [0]
          epII    ::Array{T2, 1} = [0]
          epK     ::Array{T2, 1} = [0]
          vol     ::Array{T2, 1} = [0]
    const vol_init::Array{T2, 1} = [0]
          Ms      ::Array{T2, 1} = [0]
          Mw      ::Array{T2, 1} = [0]
          Mi      ::Array{T2, 1} = [0]
          porosity::Array{T2, 1} = [0]
          cfl     ::Array{T2, 1} = [0]
    const k       ::Array{T2, 1} = [0]
          ρs      ::Array{T2, 1}
    const ρs_init ::Array{T2, 1} = [0]
          ρw      ::Array{T2, 1} = [0]
    const ρw_init ::Array{T2, 1} = [0]
    const ν       ::Array{T2, 1}
    const E       ::Array{T2, 1}
    const G       ::Array{T2, 1}
    const Ks      ::Array{T2, 1}
    const Kw      ::Array{T2, 1} = [0]
    const σt      ::Array{T2, 1} = [0] # tensile strength
    const ϕ       ::Array{T2, 1} = [0] # friction angle ϕ
    const c       ::Array{T2, 1} = [0] # cohesion
    const cr      ::Array{T2, 1} = [0] # residual cohesion
    const Hp      ::Array{T2, 1} = [0] # softening modulus
    const ψ       ::Array{T2, 1} = [0] # dilation angle ψ      
    const init    ::Array{T2, 2} = [0 0]
    const γ       ::Array{T2, 1} = [0]
    const B       ::Array{T2, 1} = [0]
          σw      ::Array{T2, 1} = [0]
          σij     ::Array{T2, 2} = [0 0]
          ϵij_s   ::Array{T2, 2} = [0 0]
          ϵij_w   ::Array{T2, 2} = [0 0]
          Δϵij_s  ::Array{T2, 2} = [0 0]
          Δϵij_w  ::Array{T2, 2} = [0 0]
          sij     ::Array{T2, 2} = [0 0]
          Vs      ::Array{T2, 2} = [0 0]
          Vw      ::Array{T2, 2} = [0 0]
          Ps      ::Array{T2, 2} = [0 0]
          Pw      ::Array{T2, 2} = [0 0]
          Ni      ::Array{T2, 2} = [0 0]
          ∂Nx     ::Array{T2, 2} = [0 0]
          ∂Ny     ::Array{T2, 2} = [0 0]
          ∂Nz     ::Array{T2, 2} = [0 0]
          ∂Fs     ::Array{T2, 2} = [0 0]
          ∂Fw     ::Array{T2, 2} = [0 0]
          F       ::Array{T2, 2} = [0 0]
    const layer   ::Array{T1, 1}
    function Particle3D{T1, T2}(num, phase, NIC, space_x, space_y, space_z, p2c, p2n, pos, 
        σm, J, epII, epK, vol, vol_init, Ms, Mw, Mi, porosity, cfl, k, ρs, ρs_init, ρw, 
        ρw_init, ν, E, G, Ks, Kw, σt, ϕ, c, cr, Hp, ψ, init, γ, B, σw, σij, ϵij_s, ϵij_w, 
        Δϵij_s, Δϵij_w, sij, Vs, Vw, Ps, Pw, Ni, ∂Nx, ∂Ny, ∂Nz, ∂Fs, ∂Fw, F, layer) where {T1, T2}
        # necessary input check
        DoF = 3
        NIC==8||NIC==64 ? nothing : error("Nodes in Cell can only be 8 or 64." )
        space_x!=T2(0)  ? nothing : error("Particle x_space is 0."             )
        space_y!=T2(0)  ? nothing : error("Particle y_space is 0."             )
        space_z!=T2(0)  ? nothing : error("Particle z_space is 0."             )
        # particles properties setup
        num      = size(pos, 1)
        vol_init = repeat([space_x*space_y*space_z], num)
        ρs_init  = copy(ρs)
        ρw_init  = copy(ρw)
        F        = repeat(T2[1 0 0 0 1 0 0 0 1] , num)
        J        = ones(T2, num)
        vol      = copy(vol_init)
        init     = copy(pos)
        phase==1 ? (num_new=1  ; DoF_new=1  ) : 
        phase==2 ? (num_new=num; DoF_new=DoF) : nothing
        p2c      = Array{T1, 1}(calloc, num             )
        Ms       = Array{T1, 1}(calloc, num             )
        Mw       = Array{T1, 1}(calloc, num_new         )
        Mi       = Array{T1, 1}(calloc, num_new         )
        epII     = Array{T2, 1}(calloc, num             )
        epK      = Array{T2, 1}(calloc, num             )
        σw       = Array{T2, 1}(calloc, num_new         )
        σm       = Array{T2, 1}(calloc, num             )
        cfl      = Array{T2, 1}(calloc, num             )
        ϵij_s    = Array{T2, 2}(calloc, num    , 6      )
        ϵij_w    = Array{T2, 2}(calloc, num_new, 6      )
        σij      = Array{T2, 2}(calloc, num    , 6      )
        Δϵij_s   = Array{T2, 2}(calloc, num    , 6      )
        Δϵij_w   = Array{T2, 2}(calloc, num_new, 6      )   
        sij      = Array{T2, 2}(calloc, num    , 6      )
        ∂Fs      = Array{T2, 2}(calloc, num    , 9      )
        ∂Fw      = Array{T2, 2}(calloc, num_new, 9      )
        p2n      = Array{T1, 2}(calloc, num    , NIC    ) 
        Ps       = Array{T2, 2}(calloc, num    , DoF    )
        Pw       = Array{T2, 2}(calloc, num_new, DoF_new)
        Vs       = Array{T2, 2}(calloc, num    , DoF    )
        Vw       = Array{T2, 2}(calloc, num_new, DoF_new)
        Ni       = Array{T2, 2}(calloc, num    , NIC    )
        ∂Nx      = Array{T2, 2}(calloc, num    , NIC    )
        ∂Ny      = Array{T2, 2}(calloc, num    , NIC    )
        ∂Nz      = Array{T2, 2}(calloc, num    , NIC    )
        # update struct
        new(num, phase, NIC, space_x, space_y, space_z, p2c, p2n, pos, σm, J, epII, epK, 
            vol, vol_init, Ms, Mw, Mi, porosity, cfl, k, ρs, ρs_init, ρw, ρw_init, ν, E, G, 
            Ks, Kw, σt, ϕ, c, cr, Hp, ψ, init, γ, B, σw, σij, ϵij_s, ϵij_w, Δϵij_s, Δϵij_w, 
            sij, Vs, Vw, Ps, Pw, Ni, ∂Nx, ∂Ny, ∂Nz, ∂Fs, ∂Fw, F, layer)
    end
end

function Base.show(io::IO, mp::PARTICLE)
    print(io, typeof(mp)                    , "\n")
    print(io, "─"^length(string(typeof(mp))), "\n")
    print(io, "particle: ", grid.node_num   , "\n")
end

#=-----------------------------------------------------------------------------------------+
| 4. Struct of VBoundary2D/3D and terminal outputs                                         |
+-----------------------------------------------------------------------------------------=#
@kwdef mutable struct VBoundary2D{T1<:Signed, T2<:AbstractFloat} <: VBC
    Vx_s_Idx::Array{T1, 1} = [0]
    Vx_s_Val::Array{T2, 1} = [0]
    Vy_s_Idx::Array{T1, 1} = [0]
    Vy_s_Val::Array{T2, 1} = [0]
    Vx_w_Idx::Array{T1, 1} = [0]
    Vx_w_Val::Array{T2, 1} = [0]
    Vy_w_Idx::Array{T1, 1} = [0]
    Vy_w_Val::Array{T2, 1} = [0]
end

@kwdef mutable struct VBoundary3D{T1<:Signed, T2<:AbstractFloat} <: VBC
    Vx_s_Idx::Array{T1, 1} = [0]
    Vx_s_Val::Array{T2, 1} = [0]
    Vy_s_Idx::Array{T1, 1} = [0]
    Vy_s_Val::Array{T2, 1} = [0]
    Vz_s_Idx::Array{T1, 1} = [0]
    Vz_s_Val::Array{T2, 1} = [0]
    Vx_w_Idx::Array{T1, 1} = [0]
    Vx_w_Val::Array{T2, 1} = [0]
    Vy_w_Idx::Array{T1, 1} = [0]
    Vy_w_Val::Array{T2, 1} = [0]
    Vz_w_Idx::Array{T1, 1} = [0]
    Vz_w_Val::Array{T2, 1} = [0]
end

function Base.show(io::IO, bc::VBC)
    print(io, typeof(bc)                    , "\n")
    print(io, "─"^length(string(typeof(bc))), "\n")
    print(io, "velocity boundary"           , "\n")
end