#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_disk.jl                                                                 |
|  Description: 2D disks contact in linear elastic constitutive equation                   |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Reference  : Sulsky, D., Chen, Z., & Schreyer, H. L. (1994). A particle method for      |
|               history-dependent materials. Computer methods in applied mechanics and     |
|               engineering, 118(1-2), 179-196.                                            |
|  Notes      : 1. Don't update the volume and density                                     |
|               2. Don't use the spin tensor                                               |
|               3. Turn off MVL                                                            |
+==========================================================================================#

using MaterialPointSolver
using CairoMakie
using ProgressMeter
using KernelAbstractions
using HDF5 

rtsdir = joinpath(homedir(), "Workbench/outputs")

@kernel inbounds=true function G2Ptest1!(
    grid::    KernelGrid2D{T1, T2},
    mp  ::KernelParticle2D{T1, T2}
) where {T1, T2}
    FNUM_0 = T2(0.0); FNUM_1 = T2(1.0)
    ix = @index(Global)
    if ix≤mp.num
        dF1 = dF2 = dF3 = dF4 = FNUM_0
        for iy in Int32(1):Int32(mp.NIC)
            p2n = mp.p2n[ix, iy]
            ∂Nx = mp.∂Nx[ix, iy]
            ∂Ny = mp.∂Ny[ix, iy]
            # compute solid incremental deformation gradient
            dF1 += grid.Δd_s[p2n, 1]*∂Nx
            dF2 += grid.Δd_s[p2n, 1]*∂Ny
            dF3 += grid.Δd_s[p2n, 2]*∂Nx
            dF4 += grid.Δd_s[p2n, 2]*∂Ny
        end
        mp.ΔFs[ix, 1] = dF1
        mp.ΔFs[ix, 2] = dF2
        mp.ΔFs[ix, 3] = dF3
        mp.ΔFs[ix, 4] = dF4
        # compute strain increment 
        mp.Δϵij_s[ix, 1] = dF1
        mp.Δϵij_s[ix, 2] = dF4
        mp.Δϵij_s[ix, 4] = dF2+dF3
        # update strain tensor
        mp.ϵij_s[ix, 1] += dF1
        mp.ϵij_s[ix, 2] += dF4
        mp.ϵij_s[ix, 4] += dF2+dF3
        # deformation gradient matrix
        F1 = mp.F[ix, 1]; F2 = mp.F[ix, 2]; F3 = mp.F[ix, 3]; F4 = mp.F[ix, 4]      
        mp.F[ix, 1] = (dF1+FNUM_1)*F1+dF2*F3
        mp.F[ix, 2] = (dF1+FNUM_1)*F2+dF2*F4
        mp.F[ix, 3] = (dF4+FNUM_1)*F3+dF3*F1
        mp.F[ix, 4] = (dF4+FNUM_1)*F4+dF3*F2
        # update jacobian value and particle volume
        mp.J[ix] = mp.F[ix, 1]*mp.F[ix, 4]-mp.F[ix, 2]*mp.F[ix, 3]
    end
end

@kernel inbounds=true function test1linear!(
    mp      ::      KernelParticle2D{T1, T2},
    pts_attr::KernelParticleProperty{T1, T2}
) where {T1, T2}
    FNUM_23 = T2(2/3); FNUM_13 = T2(1/3); FNUM_43 = T2(4/3)
    ix = @index(Global)
    if ix≤mp.num
        pid = pts_attr.layer[ix]
        Ks  = pts_attr.Ks[pid]
        G   = pts_attr.G[pid]
        # linear elastic
        Dt = Ks+FNUM_43*G
        Dd = Ks-FNUM_23*G
        mp.σij[ix, 1] += Dt*mp.Δϵij_s[ix, 1]+Dd*mp.Δϵij_s[ix, 2]+Dd*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 2] += Dd*mp.Δϵij_s[ix, 1]+Dt*mp.Δϵij_s[ix, 2]+Dd*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 3] += Dd*mp.Δϵij_s[ix, 1]+Dd*mp.Δϵij_s[ix, 2]+Dt*mp.Δϵij_s[ix, 3]
        mp.σij[ix, 4] += G *mp.Δϵij_s[ix, 4]
        # update mean stress tensor
        σm = (mp.σij[ix, 1]+mp.σij[ix, 2]+mp.σij[ix, 3])*FNUM_13
        mp.σm[ix] = σm
        # update deviatoric stress tensor
        mp.sij[ix, 1] = mp.σij[ix, 1]-σm
        mp.sij[ix, 2] = mp.σij[ix, 2]-σm
        mp.sij[ix, 3] = mp.σij[ix, 3]-σm
        mp.sij[ix, 4] = mp.σij[ix, 4]
    end
end

function test1!(args    ::MODELARGS, 
                grid    ::GRID, 
                mp      ::PARTICLE, 
                pts_attr::PROPERTY,
                bc      ::BOUNDARY,
                ΔT      ::T2,
                Ti      ::T2,
                        ::Val{:OS},
                        ::Val{:MUSL}) where {T2}
    Ti<args.Te ? G=args.gravity/args.Te*Ti : G=args.gravity
    dev = getBackend(args)
    resetgridstatus_OS!(dev)(ndrange=grid.node_num, grid)
    args.device == :CPU && args.basis == :uGIMP ? 
        resetmpstatus_OS!(grid, mp, Val(args.basis)) :
        resetmpstatus_OS!(dev)(ndrange=mp.num, grid, mp, Val(args.basis))
    P2G_OS!(dev)(ndrange=mp.num, grid, mp, G)
    solvegrid_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT, args.ζs)
    doublemapping1_OS!(dev)(ndrange=mp.num, grid, mp, pts_attr, ΔT, args.FLIP, args.PIC)
    doublemapping2_OS!(dev)(ndrange=mp.num, grid, mp)
    doublemapping3_OS!(dev)(ndrange=grid.node_num, grid, bc, ΔT)
    G2Ptest1!(dev)(ndrange=mp.num, grid, mp)
    if args.constitutive==:hyperelastic
        hyE!(dev)(ndrange=mp.num, mp, pts_attr)
    elseif args.constitutive==:linearelastic
        test1linear!(dev)(ndrange=mp.num, mp, pts_attr)
    end
    if args.MVL==true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end                                  
    return nothing
end

init_basis        = :uGIMP
init_NIC          = 16
init_phase        = 1
init_grid_space_x = 0.005#0.05
init_grid_space_y = 0.005#0.05
init_grid_range_x = [-1, 1]
init_grid_range_y = [-1, 1]
init_mp_in_space  = 2
init_project_name = "2d_disks"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :linearelastic
init_gravity      = 0
init_ζs           = 0
init_ρs           = 1000
init_ν            = 0.3
init_E            = 1e3
init_Ks           = init_E/(3(1-2*init_ν))
init_G            = init_E/(2(1+  init_ν))
init_T            = 3
init_Te           = 0
init_ΔT           = 0.5*init_grid_space_x/sqrt(init_E/init_ρs)#0.001#
init_step         = floor(init_T/init_ΔT/700) |> Int64
init_step<10 ? init_step=1 : nothing
init_scheme       = :MUSL
iInt              = Int64
iFloat            = Float64

# parameters setup
args = Args2D{iInt, iFloat}(
    Ttol         = init_T,
    Te           = init_Te, 
    ΔT           = init_ΔT,
    time_step    = :fixed,
    FLIP         = 1,
    PIC          = 0,
    ζs           = init_ζs,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = false,
    hdf5         = true,
    hdf5_step    = init_step,
    MVL          = false,
    device       = :CUDA,
    coupling     = :OS,
    scheme       = init_scheme,
    basis        = init_basis
)

# background grid setup
grid = Grid2D{iInt, iFloat}(
    range_x1 = init_grid_range_x[1],
    range_x2 = init_grid_range_x[2],
    range_y1 = init_grid_range_y[1],
    range_y2 = init_grid_range_y[2],
    space_x  = init_grid_space_x,
    space_y  = init_grid_space_y,
    NIC      = init_NIC,
    phase    = init_phase
)

# material points setup
# 1st disk
range_x = [-0.5+grid.space_x/init_mp_in_space/2, 0.5-grid.space_x/init_mp_in_space/2]
range_y = [-0.5+grid.space_y/init_mp_in_space/2, 0.5-grid.space_y/init_mp_in_space/2]
space_x = grid.space_x/init_mp_in_space
space_y = grid.space_y/init_mp_in_space
num_x   = length(range_x[1]:space_x:range_x[2])
num_y   = length(range_y[1]:space_y:range_y[2])
x_tmp   = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp   = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
mp_num  = length(x_tmp)
a = findall(i->((-0.5≤x_tmp[i]≤-0.1)&&((y_tmp[i]+0.3)^2≤(0.04-(x_tmp[i]+0.3)^2)) ||
                ( 0.1≤x_tmp[i]≤ 0.5)&&((y_tmp[i]-0.3)^2≤(0.04-(x_tmp[i]-0.3)^2))),
                1:length(x_tmp))
del_id  = deleteat!(1:mp_num |> collect, a)
map(i->splice!(i, del_id), [x_tmp, y_tmp])
mp_pos = [x_tmp y_tmp]
mp_num = length(x_tmp)
mp_ρs  = ones(mp_num).*init_ρs
mp     = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=mp_pos, ρs=mp_ρs, 
    NIC=init_NIC, phase=init_phase)
lb_id = findall(i->(-0.5≤mp.init[i, 1]≤-0.1)&&
                   ((mp.init[i, 2]+0.3)^2≤(0.04-(mp.init[i, 1]+0.3)^2)), 1:mp.num)
rt_id = findall(i->( 0.1≤mp.init[i, 1]≤ 0.5)&&
                   ((mp.init[i, 2]-0.3)^2≤(0.04-(mp.init[i, 1]-0.3)^2)), 1:mp.num)
mp.Vs[lb_id, :] .=  0.1
mp.Vs[rt_id, :] .= -0.1

# particle property setup
mp_layer = ones(mp_num)
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_Ks    = [init_Ks]
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
tmp_idx = findall(i->(grid.pos[i, 1]≤-1||grid.pos[i, 1]≥1), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 2]≤-1||grid.pos[i, 2]≥1), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num)
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc, workflow=test1!)

# analysis
@views let
    figfont = MaterialPointSolver.fontcmu
    figuretheme = Theme(
        Axis = (titlesize=30, titlefont=figfont, xticklabelfont=figfont, 
            yticklabelfont=figfont),
        Colorbar = (spinewidth=0, ticklabelfont=figfont, labelfont=figfont)
    )
    hdf5_path  = joinpath(args.project_path, "$(args.project_name).h5")
    fid        = h5open(hdf5_path, "r")
    video_step = 5
    video_fps  = 30
    itr        = (read(fid["FILE_NUM"])-1) |> Int64
    mp_pos     = Observable(fid["mp_init"] |> read)
    mp_num     = length(mp_pos[][:, 1])
    mp_Vsc     = Observable(zeros(mp_num))
    mp_σm      = Observable(zeros(mp_num))
    e1         = Observable(Point2f[(0.0, 0.0)])
    e2         = Observable(Point2f[(0.0, 0.0)])
    e3         = Observable(Point2f[(0.0, 0.0)])
    anno       = Observable(0.0)

    with_theme(figuretheme) do
        fig = Figure(size=(1080, 820), fontsize=25, font=figfont)
        Label(fig[0, 1:4], "Two elastic bodies collision", font=figfont, fontsize=30)
        ax1 = Axis(fig[1, 1], aspect=DataAspect(), xticks=(-0.5:0.2:0.5), 
            yticks=(-0.5:0.2:0.5))
        ax2 = Axis(fig[1, 3], aspect=DataAspect(), xticks=(-0.5:0.2:0.5), 
            yticks=(-0.5:0.2:0.5))
        ax3 = Axis(fig[2, 1:4], aspect=AxisAspect(4), xticks=(0:0.5:3.0), 
            yticks=(0.5:1:2.5))
        
        p1 = scatter!(ax1, mp_pos, color=mp_σm, colormap=:darktest, markersize=2, 
            colorrange=(-300, 100))
        Colorbar(fig[1, 2], p1, width=20, spinewidth=0, ticklabelfont=figfont, 
            label="Mean stress")
        p2 = scatter!(ax2, mp_pos, color=mp_Vsc, colormap=:darktest, markersize=2 ,
            colorrange=(0, 0.3))
        Colorbar(fig[1, 4], p2, width=20, spinewidth=0, ticklabelfont=figfont, 
            label="Σ Veloclty")

        p3 = scatterlines!(ax3, e1, markersize=0, linewidth=3, color=:blue )
        p4 = scatterlines!(ax3, e2, markersize=0, linewidth=3, color=:green)
        p5 = scatterlines!(ax3, e3, markersize=0, linewidth=3, color=:red  )
        axislegend(ax3, [p3, p4, p5], ["kinetic", "strain", "total"], "Energy", 
            labelsize=20, labelfont=figfont, titlefont=figfont)
        vlines!(ax3, anno  , color=:black, linewidth=1)
        vlines!(ax3, 1.5858, color=:red  , linewidth=2, linestyle=:dash)
        text!(ax3, 0.8, 1.3, text="analytical contact", font=figfont, fontsize=20)

        lines!(ax1, [0, 0], [-0.2, 0.2], color=:red, linewidth=3)
        lines!(ax1, [-0.2, 0.2], [0, 0], color=:red, linewidth=3)
        lines!(ax2, [0, 0], [-0.2, 0.2], color=:red, linewidth=3)
        lines!(ax2, [-0.2, 0.2], [0, 0], color=:red, linewidth=3)
        xlims!(ax1, -0.60, 0.6)
        ylims!(ax1, -0.60, 0.6)
        xlims!(ax2, -0.60, 0.6)
        ylims!(ax2, -0.60, 0.6)
        xlims!(ax3, -0.1, 3.5)
        ylims!(ax3, -0.1, 3.0)
        p = Progress(length(1:video_step:itr)-1; 
            desc="\e[1;36m[ Info:\e[0m $(lpad("video", 7))", color=:white, barlen=12, 
            barglyphs=BarGlyphs(" ◼◼  "))
        CairoMakie.record(fig, joinpath(args.project_path, "$(args.project_name).mp4"),
            1:video_step:itr; framerate=video_fps) do i
            mp_σij = fid["group$(i)/sig"]   |> read
            mp_Ms  = fid["group$(i)/mass"]  |> read
            mp_ϵij = fid["group$(i)/eps_s"] |> read
            mp_Vol = fid["group$(i)/vol"]   |> read
            mp_Vs  = fid["group$(i)/v_s"]   |> read
            time   = fid["group$(i)/time"]  |> read
            Vsc  = sqrt.(mp_Vs[:, 1].^2 .+mp_Vs[:, 2].^2)
            σm   = (mp_σij[:, 1].+mp_σij[:, 2].+mp_σij[:, 3])./3
            tmp1 = Point2f(time, sum(sum(0.5.*mp_Vs.^2 .*mp_Ms, dims=2)))
            tmp2 = Point2f(time, sum(0.5.*mp_σij.*mp_ϵij.*mp_Vol))
            tmp3 = Point2f(time, tmp1[2]+tmp2[2])
            mp_pos[] = fid["group$(i)/mp_pos"] |> read
            mp_Vsc[] = Vsc
            mp_σm[]  = σm
            anno[]   = time
            push!(e1[], tmp1)
            push!(e2[], tmp2)
            push!(e3[], tmp3)
            next!(p)
        end
        close(fid)
    end
end