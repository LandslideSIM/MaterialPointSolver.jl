#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : grid.jl                                                                    |
|  Description: Type system for grid in MaterialPointSolver.jl                             |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Struct     : 1. Grid2D                                                                  |
|               2. KernelGrid2D                                                            |
|               3. Grid3D                                                                  |
|               4. KernelGrid3D                                                            |
|               5. Base.show                                                               |
+==========================================================================================#

"""
    Grid2D{T1, T2}

Description:
---
This struct will save the values for 2D background grid.

The logic of tab_p2n for NIC==4 is as follows:
```julia
for i in 1:cell_num
    col_id    = cld(i, cell_num_y)       # belongs to which column
    row_id    = i-(col_id-1)*cell_num_y  # belongs to which row
    ltn_id    = col_id*node_num_y-row_id # top    left  node's index              
    c2n[i, 1] = ltn_id+1                 # bottom left  node
    c2n[i, 2] = ltn_id+0                 # top    left  node
    c2n[i, 3] = ltn_id+node_num_y+1      # bottom right node
    c2n[i, 4] = ltn_id+node_num_y        # top    right node
end
```

The logic of tab_p2n for NIC==16 is as follows:
```julia
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
```
"""
@kwdef struct Grid2D{T1, T2} <: KernelGrid2D{T1, T2}
    range_x1  ::T2
    range_x2  ::T2
    range_y1  ::T2
    range_y2  ::T2
    space_x   ::T2
    space_y   ::T2
    phase     ::T1
    node_num_x::T1 = 0
    node_num_y::T1 = 0
    node_num  ::T1 = 0
    NIC       ::T1 = 16
    pos       ::Array{T2, 2} = [0 0]
    cell_num_x::T1 = 0
    cell_num_y::T1 = 0
    cell_num  ::T1 = 0
    tab_p2n   ::Array{T1, 2} = [0 0]
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
        node_num_x, node_num_y, node_num, NIC, pos, cell_num_x, cell_num_y, cell_num, 
        tab_p2n, σm, σw, vol, Ms, Mw, Mi, Ps, Pw, Vs, Vw, Vs_T, Vw_T, Fs, Fw, Fdrag, a_s, 
        a_w, Δd_s, Δd_w) where {T1, T2}
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
            tab_p2n = T1.([0 1; 0 0; node_num_y 1; node_num_y 0])
        elseif NIC == 16
            tab_p2n = T1.([
                -node_num_y   2; -node_num_y   1; -node_num_y   0; -node_num_y   -1       
                          0   2;           0   1;           0   0;           0   -1     
                 node_num_y   2;  node_num_y   1;  node_num_y   0;  node_num_y   -1       
                 node_num_y*2 2;  node_num_y*2 1;  node_num_y*2 0;  node_num_y*2 -1 
            ])
        end
        # update struct
        new(range_x1, range_x2, range_y1, range_y2, space_x, space_y, phase, node_num_x, 
            node_num_y, node_num, NIC, pos, cell_num_x, cell_num_y, cell_num, tab_p2n, σm, 
            σw, vol, Ms, Mw, Mi, Ps, Pw, Vs, Vw, Vs_T, Vw_T, Fs, Fw, Fdrag, a_s, a_w, Δd_s, 
            Δd_w)
    end
end

"""
    struct GPUGrid2D{T1, T2, T3<:AbstractArray, 
                             T4<:AbstractArray,
                             T5<:AbstractArray}

Description:
---
Grid2D GPU struct. See [`Grid2D`](@ref) for more details.
"""
struct GPUGrid2D{T1, T2, T3<:AbstractArray, 
                         T4<:AbstractArray,
                         T5<:AbstractArray} <: KernelGrid2D{T1, T2}
    range_x1  ::T2
    range_x2  ::T2
    range_y1  ::T2
    range_y2  ::T2
    space_x   ::T2
    space_y   ::T2
    phase     ::T1
    node_num_x::T1
    node_num_y::T1
    node_num  ::T1
    NIC       ::T1
    pos       ::T5
    cell_num_x::T1
    cell_num_y::T1
    cell_num  ::T1
    tab_p2n   ::T3
    σm        ::T4
    σw        ::T4
    vol       ::T4
    Ms        ::T4
    Mw        ::T4
    Mi        ::T4
    Ps        ::T5
    Pw        ::T5
    Vs        ::T5
    Vw        ::T5
    Vs_T      ::T5
    Vw_T      ::T5
    Fs        ::T5
    Fw        ::T5
    Fdrag     ::T5
    a_s       ::T5
    a_w       ::T5
    Δd_s      ::T5
    Δd_w      ::T5
end

"""
    Grid3D{T1, T2} <: GRID

Description:
---
This struct will save the values for 3D background grid.

The logic of tab_p2n for NIC==8 is as follows:
```julia
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
```

The logic of tab_p2n for NIC==64 is as follows:
```julia
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
    end
end
```
"""
@kwdef struct Grid3D{T1, T2} <: KernelGrid3D{T1, T2}
    range_x1  ::T2
    range_x2  ::T2
    range_y1  ::T2
    range_y2  ::T2
    range_z1  ::T2
    range_z2  ::T2
    space_x   ::T2
    space_y   ::T2
    space_z   ::T2
    phase     ::T1
    node_num_x::T1 = 0
    node_num_y::T1 = 0
    node_num_z::T1 = 0
    node_num  ::T1 = 0
    NIC       ::T1 = 64
    pos       ::Array{T2, 2} = [0 0]
    cell_num_x::T1 = 0
    cell_num_y::T1 = 0
    cell_num_z::T1 = 0
    cell_num  ::T1 = 0
    tab_p2n   ::Array{T1, 2} = [0 0]
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
        pos, cell_num_x, cell_num_y, cell_num_z, cell_num, tab_p2n, σm, σw, vol, Ms, Mw, Mi, 
        Ps, Pw, Vs, Vw, Vs_T, Vw_T, Fs, Fw, Fdrag, a_s, a_w, Δd_s, Δd_w) where {T1, T2}
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
            tab_p2n = T1.([
                -1 -1 -1; -1 0 -1; 0 0 -1; 0 -1 -1
                -1 -1  0; -1 0  0; 0 0  0; 0 -1  0
            ])
        elseif NIC==64
            tab_p2n = T1.([
                -1 -1 -1 +0; -1 -0 -1 +0; -0 -0 -1 +0; -0 -1 -1 +0; -0 -2 -1 +0; -1 -2 -1 +0 
                -2 -2 -1 +0; -2 -1 -1 +0; -2 -0 -1 +0; -2 +1 -1 +0; -1 +1 -1 +0; -0 +1 -1 +0 
                +1 +1 -1 +0; +1 -0 -1 +0; +1 -1 -1 +0; +1 -2 -1 +0; -1 -1 -1 +1; -1 -0 -1 +1 
                -0 -0 -1 +1; -0 -1 -1 +1; -0 -2 -1 +1; -1 -2 -1 +1; -2 -2 -1 +1; -2 -1 -1 +1 
                -2 -0 -1 +1; -2 +1 -1 +1; -1 +1 -1 +1; -0 +1 -1 +1; +1 +1 -1 +1; +1 -0 -1 +1 
                +1 -1 -1 +1; +1 -2 -1 +1; -1 -1 -0 +1; -1 -0 -0 +1; -0 -0 -0 +1; -0 -1 -0 +1 
                -0 -2 -0 +1; -1 -2 -0 +1; -2 -2 -0 +1; -2 -1 -0 +1; -2 -0 -0 +1; -2 +1 -0 +1 
                -1 +1 -0 +1; -0 +1 -0 +1; +1 +1 -0 +1; +1 -0 -0 +1; +1 -1 -0 +1; +1 -2 -0 +1 
                -1 -1 -0 +2; -1 -0 -0 +2; -0 -0 -0 +2; -0 -1 -0 +2; -0 -2 -0 +2; -1 -2 -0 +2 
                -2 -2 -0 +2; -2 -1 -0 +2; -2 -0 -0 +2; -2 +1 -0 +2; -1 +1 -0 +2; -0 +1 -0 +2 
                +1 +1 -0 +2; +1 -0 -0 +2; +1 -1 -0 +2; +1 -2 -0 +2
            ])
        end
        # update struct
        new(range_x1, range_x2, range_y1, range_y2, range_z1, range_z2, space_x, space_y, 
            space_z, phase, node_num_x, node_num_y, node_num_z, node_num, NIC, pos, 
            cell_num_x, cell_num_y, cell_num_z, cell_num, tab_p2n, σm, σw, vol, Ms, Mw, Mi, 
            Ps, Pw, Vs, Vw, Vs_T, Vw_T, Fs, Fw, Fdrag, a_s, a_w, Δd_s, Δd_w)
    end
end

"""
    struct GPUGrid3D{T1, T2, T3<:AbstractArray, 
                             T4<:AbstractArray,
                             T5<:AbstractArray}

Description:
---
Grid3D GPU struct. See [`Grid3D`](@ref) for more details.
"""
struct GPUGrid3D{T1, T2, T3<:AbstractArray, 
                         T4<:AbstractArray,
                         T5<:AbstractArray} <: KernelGrid3D{T1, T2}
    range_x1  ::T2
    range_x2  ::T2
    range_y1  ::T2
    range_y2  ::T2
    range_z1  ::T2
    range_z2  ::T2
    space_x   ::T2
    space_y   ::T2
    space_z   ::T2
    phase     ::T1
    node_num_x::T1
    node_num_y::T1
    node_num_z::T1
    node_num  ::T1
    NIC       ::T1
    pos       ::T5
    cell_num_x::T1
    cell_num_y::T1
    cell_num_z::T1
    cell_num  ::T1
    tab_p2n   ::T3
    σm        ::T4
    σw        ::T4
    vol       ::T4
    Ms        ::T4
    Mw        ::T4
    Mi        ::T4
    Ps        ::T5
    Pw        ::T5
    Vs        ::T5
    Vw        ::T5
    Vs_T      ::T5
    Vw_T      ::T5
    Fs        ::T5
    Fw        ::T5
    Fdrag     ::T5
    a_s       ::T5
    a_w       ::T5
    Δd_s      ::T5
    Δd_w      ::T5
end

function Base.show(io::IO, grid::GRID)
    print(io, typeof(grid)                    , "\n")
    print(io, "─"^length(string(typeof(grid))), "\n")
    print(io, "node: ", grid.node_num         , "\n")
    print(io, "cell: ", grid.cell_num         , "\n")
end