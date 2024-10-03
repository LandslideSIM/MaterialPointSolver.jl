#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : modelinfo.jl                                                               |
|  Description: Export or import model in JSON format                                      |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. check_datasize                                                         |
|               02. @memcheck                                                              |
+==========================================================================================#

export check_datasize, @memcheck

function check_datasize(args::     DeviceArgs{T1, T2}, 
                        grid::     DeviceGrid{T1, T2},
                        mp  :: DeviceParticle{T1, T2},
                        attr:: DeviceProperty{T1, T2},
                        bc  ::DeviceVBoundary{T1, T2}) where {T1, T2}
    args_mem  = Base.summarysize(args)
    grid_mem  = Base.summarysize(grid)
    mp_mem    = Base.summarysize(mp)
    attr_mem  = Base.summarysize(attr)
    bc_mem    = Base.summarysize(bc)
    mem_total = args_mem + grid_mem + mp_mem + attr_mem + bc_mem

    args_size = lpad(@sprintf("%.2f", args_mem / 1024 ^ 3), 5)
    grid_size = lpad(@sprintf("%.2f", grid_mem / 1024 ^ 3), 5)
    mp_size   = lpad(@sprintf("%.2f", mp_mem   / 1024 ^ 3), 5)
    attr_size = lpad(@sprintf("%.2f", attr_mem / 1024 ^ 3), 5)
    bc_size   = lpad(@sprintf("%.2f", bc_mem   / 1024 ^ 3), 5)
    
    argsp  = lpad(@sprintf("%.2f", args_mem / mem_total * 100), 5)
    gridp  = lpad(@sprintf("%.2f", grid_mem / mem_total * 100), 5)
    mpp    = lpad(@sprintf("%.2f", mp_mem   / mem_total * 100), 5)
    attr_p = lpad(@sprintf("%.2f", attr_mem / mem_total * 100), 5)
    bcp    = lpad(@sprintf("%.2f", bc_mem   / mem_total * 100), 5)

    tbar = string("─"^19, "┬", "─"^11, "┬", "─"^8)
    bbar = string("─"^19, "┴", "─"^11, "┴", "─"^8)
    @info """model data size
    $(tbar)
    MPM model args     │ $args_size GiB │ $argsp %
    background grid    │ $grid_size GiB │ $gridp %
    material particles │ $mp_size GiB │ $mpp %
    particle properties│ $attr_size GiB │ $attr_p %
    boundary conditions│ $bc_size GiB │ $bcp %
    $(bbar)
    """
    return nothing
end

macro memcheck(expr)
    return quote        
        data = Base.summarysize($(esc(expr))) / 1024 ^ 3
        data_str = @sprintf("%.2f GiB", data)
        @info "💾 $data_str" # print info
    end
end
