#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : p2nindex.jl                                                                |
|  Description: The index of the node corresponding to the given cell index in the grid.   |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. getP2N_linear [2D & 3D]                                                 |
|               2. getP2N_uGIMP [2D & 3D]                                                  |
+==========================================================================================#

export getP2N_linear, getP2N_uGIMP

"""
    getP2N_linear(grid::KernelGrid2D{T1, T2}, p2c::T1, iy::Int32)

Description:
---
Get the index of the node corresponding to the given cell index in the linear basis grid [2D].

```julia
col_id    = cld(i, cell_num_y)       # belongs to which column
row_id    = i-(col_id-1)*cell_num_y  # belongs to which row
ltn_id    = col_id*node_num_y-row_id # top    left  node's index              
c2n[i, 1] = ltn_id+1                 # bottom left  node
c2n[i, 2] = ltn_id+0                 # top    left  node
c2n[i, 3] = ltn_id+node_num_y+1      # bottom right node
c2n[i, 4] = ltn_id+node_num_y        # top    right node

tab_p2n = T1.([0 1; 0 0; node_num_y 1; node_num_y 0])
````
"""
@inline function getP2N_linear(
    grid::KernelGrid2D{T1, T2}, 
    p2c ::T1, 
    iy  ::Int32
) where {T1, T2}
    col_id = cld(p2c, grid.cell_num_y)            # belongs to which column
    row_id = p2c - (col_id - 1) * grid.cell_num_y # belongs to which row
    ltn_id = col_id * grid.node_num_y - row_id    # top left node's index  

    return (ltn_id + grid.tab_p2n[iy, 1] + grid.tab_p2n[iy, 2]) |> Int32
end

"""
    getP2N_linear(grid::KernelGrid3D{T1, T2}, p2c::T1, iy::Int32)

Description:
---
Get the index of the node corresponding to the given cell index in the linear basis grid [3D].

```julia
for i in 1:cell_num
    layer = cld(i, cell_num_x*cell_num_y) # 处于第几层
    l_idx = cld(i-(layer-1)*cell_num_x*cell_num_y, cell_num_y) # 处于第几行
    l_idy = (i-(layer-1)*cell_num_x*cell_num_y)-(l_idx-1)*cell_num_y # 处于第几列

    c2n[i, 1] = (layer-1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+1
    c2n[i, 2] = (layer-1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+1
    c2n[i, 3] = (layer-0)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-1)+1
    c2n[i, 4] = (layer-0)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-1)+1
    c2n[i, 5] = (layer-1)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+1
    c2n[i, 6] = (layer-1)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+1
    c2n[i, 7] = (layer-0)*node_num_x*node_num_y+(l_idx-0)*node_num_y+(l_idy-0)+1
    c2n[i, 8] = (layer-0)*node_num_x*node_num_y+(l_idx-1)*node_num_y+(l_idy-0)+1
end

tab_p2n = T1.([
    -1 -1 -1; -1 0 -1; 0 0 -1; 0 -1 -1
    -1 -1  0; -1 0  0; 0 0  0; 0 -1  0
])
````
"""
@inline function getP2N_linear(
    grid::KernelGrid3D{T1, T2}, 
    p2c ::T1, 
    iy  ::Int32
) where {T1, T2}
    layer = cld(p2c, grid.cell_num_x*grid.cell_num_y)                                 # 处于第几层
    l_idx = cld(p2c-(layer-1)*grid.cell_num_x*grid.cell_num_y, grid.cell_num_y)       # 处于第几行
    l_idy = (p2c-(layer-1)*grid.cell_num_x*grid.cell_num_y)-(l_idx-1)*grid.cell_num_y # 处于第几列

    return ((layer + grid.tab_p2n[iy, 1]) * grid.node_num_x * grid.node_num_y +
            (l_idx + grid.tab_p2n[iy, 2]) * grid.node_num_y +
            (l_idy + grid.tab_p2n[iy, 3]) + 1) |> Int32
end

"""
    getP2N_uGIMP(grid::KernelGrid2D{T1, T2}, p2c::T1, iy::Int32)

Description:
---
Get the index of the node corresponding to the given cell index in the uGIMP basis grid [2D].

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

tab_p2n = T1.([
    -node_num_y   2; -node_num_y   1; -node_num_y   0; -node_num_y   -1       
              0   2;           0   1;           0   0;           0   -1     
     node_num_y   2;  node_num_y   1;  node_num_y   0;  node_num_y   -1       
     node_num_y*2 2;  node_num_y*2 1;  node_num_y*2 0;  node_num_y*2 -1 
])
````
"""
@inline function getP2N_uGIMP(
    grid::KernelGrid2D{T1, T2}, 
    p2c ::T1, 
    iy  ::Int32
) where {T1, T2}
    col_id = cld(p2c, grid.cell_num_y)            # belongs to which column
    row_id = p2c - (col_id - 1) * grid.cell_num_y # belongs to which row
    ltn_id = col_id * grid.node_num_y - row_id    # top left node's index    

    return (ltn_id+grid.tab_p2n[iy, 1]+grid.tab_p2n[iy, 2]) |> Int32
end

"""
    getP2N_uGIMP(grid::KernelGrid3D{T1, T2}, p2c::T1, iy::Int32)

Description:
---
Get the index of the node corresponding to the given cell index in the uGIMP basis grid [3D].

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
```
"""
@inline function getP2N_uGIMP(
    grid::KernelGrid3D{T1, T2}, 
    p2c ::T1, 
    iy  ::Int32
) where {T1, T2}
    layer = cld(p2c, grid.cell_num_x*grid.cell_num_y)                                 
    l_idx = cld(p2c-(layer-1)*grid.cell_num_x*grid.cell_num_y, grid.cell_num_y)       
    l_idy = (p2c-(layer-1)*grid.cell_num_x*grid.cell_num_y)-(l_idx-1)*grid.cell_num_y

    return ((layer + grid.tab_p2n[iy, 1]) * grid.node_num_x * grid.node_num_y +
            (l_idx + grid.tab_p2n[iy, 2]) * grid.node_num_y +
            (l_idy + grid.tab_p2n[iy, 3]) + grid.tab_p2n[iy, 4]) |> Int32
end
