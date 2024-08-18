#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : terminaltxt.jl                                                             |
|  Description: Contents will print in terminal                                            |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 1. info_print                                                              |
|               2. perf                                                                    |
|               3. format_seconds                                                          |
|               4. progressinfo                                                            |
|               5. updatepb!                                                               |
+==========================================================================================#

"""
    info_print(args::MODELARGS, grid::GRID, mp::PARTICLE)

Description:
---
Print the simulation info.
"""
function info_print(args::MODELARGS, grid::GRID, mp::PARTICLE)
    pc = typeof(args).parameters[2]==Float64 ? "FP64" : "FP32"
    di = typeof(args) <: Args2D ? "2D/" : "3D/"
    ΔT = args.time_step==:fixed ? string(@sprintf("%.2e", args.ΔT), "s") : "adaptive "
    ct = string(di, "$(args.device)")
    args.constitutive==:linearelastic ? material="L-E" :
    args.constitutive==:hyperelastic  ? material="H-E" :
    args.constitutive==:druckerprager ? material="D-P" :
    args.constitutive==:mohrcoulomb   ? material="M-C" : material="U-D"
    text_place1 = 5
    text_place2 = 9
    pic  = lpad(       @sprintf("%.2f", args.PIC           ), text_place1)
    flip = lpad(       @sprintf("%.2f", args.FLIP          ), text_place1)
    ζs   = lpad(       @sprintf("%.2f", args.ζs            ), text_place1)
    ζw   = lpad(       @sprintf("%.2f", args.ζw            ), text_place1)
    hd   = lpad(string(                 args.hdf5          ), text_place1)
    ΔT   = lpad(                        ΔT                  , text_place2)
    Ttol = lpad(string(@sprintf("%.2e", args.Ttol    ), "s"), text_place2)
    pts  = lpad(string(@sprintf("%.2e", mp.num       ), " "), text_place2)
    nds  = lpad(string(@sprintf("%.2e", grid.node_num), " "), text_place2)
    mvl  = lpad(string(                 args.MVL      , " "), text_place2)
    @info """$(args.project_name) [$(ct)]
    ────────────────┬─────────────┬─────────────────
    ΔT  : $(ΔT) │ PIC : $(pic) │ scheme   : $(args.scheme)
    Ttol: $(Ttol) │ FLIP: $(flip) │ coupling : $(args.coupling)
    pts : $(pts) │ ζs  : $(ζs) │ animation: $(args.animation)
    nds : $(nds) │ ζw  : $(ζw) │ precision: $(pc)
    MVL : $(mvl) │ HDF5: $(hd) │ material : $(material)
    ────────────────┴─────────────┴─────────────────
    """
    return nothing
end

"""
    perf(args::MODELARGS)

Description:
---
Print the performance summary.
"""
function perf(args::MODELARGS, grid::GRID, mp::PARTICLE)
    its = @sprintf "%.2e" args.iter_num / (args.end_time - args.start_time)
    iter_num = @sprintf "%.2e" args.iter_num
    wtime = format_seconds(args.end_time - args.start_time)
    DoF = typeof(args)<:Args2D ? 2 : 
          typeof(args)<:Args3D ? 3 : nothing
    nio = ((grid.node_num + grid.node_num * DoF * 4) * 2 + grid.node_num * DoF + 
            grid.cell_num * mp.NIC) + ((8 * mp.num + 11 * mp.num * DoF + 
            mp.num * mp.NIC * 2 + mp.num * mp.NIC * DoF + 
            2 * mp.num * (DoF * DoF)) * 2 + mp.num)
    args.MVL==true ? nio += ((grid.cell_num * 2) * 2) : nothing
    mteff = @sprintf "%.2e" (nio * sizeof(eltype(args.Ttol)) * args.iter_num) / (
        (args.end_time - args.start_time) * (1024 ^ 3))
    # print info
    l1 = length("┌ Info: performance") - 2
    l2 = length("iters: $(iter_num)")
    l3 = length("wtime: $(wtime)")
    l4 = length("speed: $(its) it/s")
    if args.coupling==:OS && args.device != :CPU
        l5 = length("MTeff: $(mteff) GiB/s")
        bar = "─" ^ max(l1, l2, l3, l4, l5)
        @info """performance
        $(bar)
        wtime: $(wtime)
        iters: $(iter_num)
        speed: $(its)  it/s
        MTeff: $(mteff) GiB/s
        $(bar)
        """
    else
        bar = "─" ^ max(l1, l2, l3, l4)
        @info """performance
        $(bar)
        wtime: $(wtime)
        iters: $(iter_num)
        speed: $(its) it/s
        $(bar)
        """
    end
    return nothing
end

"""
    format_seconds(s)

Description:
---
Format seconds to days, hours, minutes, and seconds.
"""
function format_seconds(s)
    s<1 ? s=1 : nothing
    s    = ceil(s)
    dt   = Dates.Second(s)  
    days = Dates.value(dt)÷(60*60*24)
    days==0 ? (
        time = Dates.Time(Dates.unix2datetime(Dates.value(dt) % (60 * 60 * 24)));
        return string(Dates.format(time, "HH:MM:SS"))                           ;
    ) : (
        time = Dates.Time(Dates.unix2datetime(Dates.value(dt) % (60 * 60 * 24)))    ;
        return string(lpad(days, 2, "0"), " days: ", Dates.format(time, "HH:MM:SS"));
    )
end

"""
    progressinfo(words::String; dt=3.0)

Description:
---
Print the progress info.
"""
function progressinfo(args::MODELARGS, words::String; dt=3.0)
    return Progress(100, dt=dt; desc="\e[1;36m[ Info:\e[0m $(lpad(words, 7))",
        color=:white, barlen=12, barglyphs=BarGlyphs(" ◼◼  "), output=stderr, 
        enabled=args.progressbar)
end

function updatepb!(pc::Ref{T1}, Ti::T2, Ttol::T2, p::Progress) where {T1, T2}
    pos = trunc(Ti/Ttol*100) |> Int64
    pos≥100 ? (pos=100; pc[]+=1) : nothing
    pc[]≤1 ? update!(p, pos)     : nothing
    return nothing
end