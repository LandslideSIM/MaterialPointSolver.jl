#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : toolkit.jl                                                                 |
|  Description: Some extra cpu tools for MPMSolver.jl                                      |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Functions  : 01. info_page()                                                            |
|               02. info_print()                                                           |
|               03. perf()                                                                 |
|               04. format_seconds()                                                       |
|               05. progressinfo()                                                         |
|               06. updatepb!()                                                            |
|               07. cfl() [2D OS]                                                          |
|               08. cfl() [3D OS]                                                          |
|               09. cfl() [2D TS]                                                          |
|               10. cfl() [3D TS]                                                          |
|               11. model_info()                                                           |
+==========================================================================================#

"""
    info_page()

Description:
---
Welcome logo!
"""
function info_page()
    print("\e[0;0H\e[2J")
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       __  ______  __  _______     __                 _ __     ║
    ║      /  |/  / _ \\/  |/  / __/__  / /  _____ ____    (_) /     ║
    ║     / /|_/ / ___/ /|_/ /\\ \\/ _ \\/ / |/ / -_) __/   / / /      ║
    ║    /_/  /_/_/  /_/  /_/___/\\___/_/|___/\\__/_/ (_)_/ /_/       ║
    ║                                                  |___/        ║
    ║    $("─"^54)     ║
    """)
    print("║ "); printstyled("        High-performance MPM Solver for Geomechanics",
        color=:cyan, bold=true); print("$(" "^10)║", "\n")
    print("║ "); printstyled(" ⦿  Programmer :" , color=:blue , bold=true)
    print(" 霍泽楠 - Zenan Huo"); print("$(" "^27)║", "\n")
    print("║ "); printstyled(" ⦿  Start Date :" , color=:red  , bold=true)
    print(" 01/01/2022"); print("$(" "^35)║", "\n")
    print("║ "); printstyled(" ⦿  Affiliation:", color=:green, bold=true)
    print(" Risk Group, UNIL-ISTE"); print("$(" "^24)║", "\n")
    print("""╚═══════════════════════════════════════════════════════════════╝\n\n""")
end

"""
    info_print(args::ARGS, mp::PARTICLE)

Description:
---
Print the simulation info.
"""
function info_print(args::ARGS, mp::PARTICLE)
    typeof(args) <: Args2D ? di = "2D/" : di = "3D/"
    dev = string(args.device)
    coupling = string(args.coupling)
    args.constitutive==:linearelastic ? ct=*(di, "L-E", "/$(coupling)", "/$(dev)") :
    args.constitutive==:hyperelastic  ? ct=*(di, "H-E", "/$(coupling)", "/$(dev)") :
    args.constitutive==:druckerprager ? ct=*(di, "D-P", "/$(coupling)", "/$(dev)") :
    args.constitutive==:mohrcoulomb   ? ct=*(di, "M-C", "/$(coupling)", "/$(dev)") : 
    args.constitutive==:taitwater     ? ct=*(di, "T-W", "/$(coupling)", "/$(dev)") : nothing
    if args.time_step == :fixed
        ΔT   = @sprintf("%.2e", args.ΔT)
        Ttol = @sprintf("%.2e", args.Ttol)
        pic  = @sprintf("%.2f", args.PIC)
        flip = @sprintf("%.2f", args.FLIP)
        ζ    = @sprintf("%.2f", args.ζ)
        ps   = @sprintf("%.2e", mp.num)
        @info """$(args.project_name) ($(ct))
        ──────────────────┬────────────┬─────────────────
        ΔT  : $(lpad(ΔT, 10))s │ PIC : $(pic) │ HDF5     : $(args.hdf5)
        Ttol: $(lpad(Ttol, 10))s │ FLIP: $(flip) │ animation: $(args.animation)
        pts : $(lpad(ps, 10))  │ ζ   : $(ζ) │ vollock  : $(args.vollock)
        ──────────────────┴────────────┴─────────────────
        """
    elseif args.time_step == :auto
        ΔT   = @sprintf("%.2e", args.ΔT)
        Ttol = @sprintf("%.2e", args.Ttol)
        pic  = @sprintf("%.2f", args.PIC)
        flip = @sprintf("%.2f", args.FLIP)
        ζ    = @sprintf("%.2f", args.ζ)
        ps   = @sprintf("%.2e", mp.num)
        @info """MPM Configuration ($(ct))
        ──────────────────┬────────────┬───────────────
        ΔT  : $(lpad("adaptive", 10))  │ PIC : $(pic) │ HDF5   : $(args.hdf5)
        Ttol: $(lpad(Ttol, 10))s │ FLIP: $(flip) │ VTK    : $(args.animation)
        pts : $(lpad(ps, 10))  │ ζ   : $(ζ) │ vollock: $(args.vollock)
        ──────────────────┴────────────┴───────────────
        """
    end
    return nothing
end

"""
    perf(args::ARGS)

Description:
---
Print the performance summary.
"""
function perf(args::ARGS, grid::GRID, mp::PARTICLE)
    its      = @sprintf "%.2e" args.iter_num/(args.end_time-args.start_time)
    iter_num = @sprintf "%.2e" args.iter_num
    wtime    = format_seconds(args.end_time-args.start_time)
    typeof(args)<:Args2D ? DoF=2 : 
    typeof(args)<:Args3D ? DoF=3 : nothing
    nio = ((grid.node_num+grid.node_num*DoF*5)*2+grid.node_num*DoF+grid.cell_num*mp.NIC)+
          ((8*mp.num+11*mp.num*DoF+mp.num*mp.NIC*2+mp.num*mp.NIC*DoF+2*mp.num*(DoF^2))*2+mp.num)
    args.vollock==true ? nio+=((grid.cell_num*2)*2) : nothing
    mteff = @sprintf "%.2e" (nio*sizeof(eltype(args.Ttol))*args.iter_num)/(
        (args.end_time-args.start_time)*(1024^3))
    # print info
    l1 = length("┌ Info: performance")-2
    l2 = length("iters: $(iter_num)")
    l3 = length("wtime: $(wtime)")
    l4 = length("speed: $(its) it/s")
    if args.coupling==:OS
        l5 = length("MTeff: $(mteff) GiB/s")
        bar = "─"^max(l1, l2, l3, l4, l5)
        @info """performance
        $(bar)
        wtime: $(wtime)
        iters: $(iter_num)
        speed: $(its)  it/s
        MTeff: $(mteff) GiB/s
        $(bar)
        """
    elseif args.coupling==:TS
        bar = "─"^max(l1, l2, l3, l4)
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
function progressinfo(args::ARGS, words::String; dt=3.0)
    return Progress(100, dt=dt; desc="\e[1;36m[ Info:\e[0m $(lpad(words, 7))",
        color=:white, barlen=12, barglyphs=BarGlyphs(" ◼◼  "), output=stderr, 
        enabled=args.progressbar)
end

function updatepb!(idc::Ref{T1}, Ti::T2, Ttol::T2, p::Progress) where {T1, T2}
    pos = trunc(Ti/Ttol*100)|> Int64
    pos≥100 ? (pos=100; idc[]+=1) : nothing
    idc[]≤1 ? update!(p, pos)     : nothing
    return nothing
end

"""
    cfl(args::Args2D{T1, T2}, grid::Grid2D{T1, T2}, mp::Particle2D{T1, T2}, ::Val{:OS})

Description:
---
Calculate the CFL condition for 2D simulation (one-phase single point).
"""    
function cfl(args::    Args2D{T1, T2}, 
             grid::    Grid2D{T1, T2}, 
             mp  ::Particle2D{T1, T2},
                 ::Val{:OS}) where {T1, T2}
    ΔT      = T2(0.0)
    FNUM_43 = T2(4/3) 
    for i in 1:mp.num
        pid       = mp.layer[i]
        Ks        = mp.Ks[pid]
        G         = mp.G[pid]
        sqr       = sqrt((Ks+G*FNUM_43)/mp.ρs[i])
        cd_sx     = grid.space_x/(sqr+abs(mp.Vs[i, 1]))
        cd_sy     = grid.space_y/(sqr+abs(mp.Vs[i, 2]))
        val       = min(cd_sx, cd_sy)
        mp.cfl[i] = args.αT*val
    end
    return ΔT = minimum(mp.cfl)
end

"""
    cfl(args::Args3D{T1, T2}, grid::Grid3D{T1, T2}, mp::Particle3D{T1, T2}, ::Val{:OS})

Description:
---
Calculate the CFL condition for 3D simulation (one-phase single point).
"""
function cfl(args::    Args3D{T1, T2}, 
             grid::    Grid3D{T1, T2}, 
             mp  ::Particle3D{T1, T2},
                 ::Val{:OS}) where {T1, T2}
    ΔT      = T2(0.0)
    FNUM_43 = T2(4/3) 
    for i in 1:mp.num
        pid       = mp.layer[i]
        Ks        = mp.Ks[pid]
        G         = mp.G[pid]
        sqr       = sqrt((Ks+G*FNUM_43)/mp.ρs[i])
        cd_sx     = grid.space_x/(sqr+abs(mp.Vs[i, 1]))
        cd_sy     = grid.space_y/(sqr+abs(mp.Vs[i, 2]))
        cd_sz     = grid.space_z/(sqr+abs(mp.Vs[i, 3]))
        val       = min(cd_sx, cd_sy, cd_sz)
        mp.cfl[i] = args.αT*val
    end
    return ΔT = minimum(mp.cfl)
end

"""
    cfl(args::Args2D{T1, T2}, grid::Grid2D{T1, T2}, mp::Particle2D{T1, T2}, ::Val{:TS})

Description:
---
Calculate the CFL condition for 2D simulation (two-phase single point).
"""
function cfl(args::    Args2D{T1, T2}, 
             grid::    Grid2D{T1, T2}, 
             mp  ::Particle2D{T1, T2},
                 ::Val{:TS}) where {T1, T2}
    FNUM_1 = T2(1.0)
    ΔT     = T2(0.0)
    for i in 1:mp.num
        sqr = sqrt((mp.E[i]+mp.Kw[i]*mp.porosity[i])/(mp.ρs[i]*(FNUM_1-mp.porosity[i])+
                                                      mp.ρw[i]*        mp.porosity[i]))
        cd_sx = grid.space_x/(sqr+abs(mp.Vs[i, 1]))
        cd_sy = grid.space_y/(sqr+abs(mp.Vs[i, 2]))
        cd_wx = grid.space_x/(sqr+abs(mp.Vw[i, 1]))
        cd_wy = grid.space_y/(sqr+abs(mp.Vw[i, 2]))
        val   = min(cd_sx, cd_sy, cd_wx, cd_wy)
        mp.cfl[i] = args.αT*val
    end
    return ΔT = minimum(mp.cfl)
end

"""
    cfl(args::Args3D{T1, T2}, grid::Grid3D{T1, T2}, mp::Particle3D{T1, T2}, ::Val{:TS})

Description:
---
Calculate the CFL condition for 3D simulation (two-phase single point).
"""
function cfl(args::    Args3D{T1, T2}, 
             grid::    Grid3D{T1, T2}, 
             mp  ::Particle3D{T1, T2},
                 ::Val{:TS}) where {T1, T2}
    FNUM_1 = T2(1.0)
    ΔT     = T2(0.0)
    for i in 1:mp.num
        sqr = sqrt((mp.E[i]+mp.Kw[i]*mp.porosity[i])/(mp.ρs[i]*(FNUM_1-mp.porosity[i])+
                                                      mp.ρw[i]*        mp.porosity[i]))
        cd_sx = grid.space_x/(sqr+abs(mp.Vs[i, 1]))
        cd_sy = grid.space_y/(sqr+abs(mp.Vs[i, 2]))
        cd_sz = grid.space_z/(sqr+abs(mp.Vs[i, 3]))
        cd_wx = grid.space_x/(sqr+abs(mp.Vw[i, 1]))
        cd_wy = grid.space_y/(sqr+abs(mp.Vw[i, 2]))
        cd_wz = grid.space_z/(sqr+abs(mp.Vw[i, 3]))
        val   = min(cd_sx, cd_sy, cd_sz, cd_wx, cd_wy, cd_wz)
        mp.cfl[i] = args.αT*val
    end
    return ΔT = minimum(mp.cfl)
end

"""
    model_info(args::ARGS, grid::GRID, mp::PARTICLE)

Description:
---
Export model's configuration information into json file.
"""
function model_info(args::ARGS, grid::GRID, mp::PARTICLE)
    file = open(joinpath(args.project_path, "model_info.txt"), "w")
    leftsize = maximum(length.(string.(fieldnames(typeof(args)))))+12
    original_stdout = stdout
    redirect_stdout(file)
    try
        println("Model Configuration:")
        println("====================")
        for item in fieldnames(typeof(args))
            dash_count = leftsize-length(string(item))
            dashes = repeat("─", dash_count)
            println("$(string.(item)) $dashes $(getfield(args, item))")
        end
        println()
        dashes = repeat("─", leftsize-length("mp_space_x"))
        println("Particle Configuration:")
        println("=======================")
        println("mp_space_x $dashes $(mp.space_x)")
        println("mp_space_y $dashes $(mp.space_y)")
        typeof(args)<:Args3D ? println("mp_space_z $dashes $(mp.space_z)") : nothing
        dashes = repeat("─", leftsize-length("mp_num"))
        println("mp_num $dashes $(mp.num)")
        dashes = repeat("─", leftsize-length("NIC"))
        println("NIC $dashes $(mp.NIC)")
        println()
        dashes = repeat("─", leftsize-length("grid_space_x"))
        println("Grid Configuration:")
        println("===================")
        println("grid_space_x $dashes $(grid.space_x)")
        println("grid_space_y $dashes $(grid.space_y)")
        typeof(args)<:Args3D ? println("grid_space_z $dashes $(grid.space_z)") : nothing
        dashes = repeat("─", leftsize-length("node_num"))
        println("node_num $dashes $(grid.node_num)")
        dashes = repeat("─", leftsize-length("cell_num"))
        println("cell_num $dashes $(grid.cell_num)")
    finally
        redirect_stdout(original_stdout)
        close(file)
    end
    @info "model info exported to project path"
    return nothing
end