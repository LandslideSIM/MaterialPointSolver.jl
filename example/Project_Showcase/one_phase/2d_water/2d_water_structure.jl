#==========================================================================================+
|           MaterialPointSolver.jl: High-performance MPM Solver for Geomechanics           |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_water_structure.jl                                                      |
|  Description: Verification example: dam break against an elastic obstacle                |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Reference  : [1] Numerical modeling of friction in lubricated cold rolling, DOCTORAL    |
|                   THESIS, https://orbi.uliege.be/handle/2268/245441 (Page 311)           |
|               [2] DATA SOURCE: Mitsume, Naoto. "Compatible interface wave–structure      |
|                   interaction model for combining mesh-free particle and finite element  |
|                   methods." Advanced Modeling and Simulation in Engineering Sciences 10, |
|                   no. 1 (2023): 11.                                                      |
|               [3] FLUID SIMULATION EQUATIONS: Meduri, S., Cremonesi, M., Perego, U.,     |
|                   Bettinotti, O., Kurkchubasche, A., & Oancea, V. (2018). A partitioned  |
|                   fully explicit Lagrangian finite element method for highly nonlinear   |
|                   fluid‐structure interaction problems. International Journal for        |
|                   Numerical Methods in Engineering, 113(1), 43-64.                       |
|  Note       : This example may break under high resolution                               |
|               This example still has some problems                                       |
+==========================================================================================#

using CairoMakie
using DelimitedFiles
using HDF5
using MaterialPointSolver
using KernelAbstractions

rtsdir = joinpath(homedir(), "Workbench/outputs")

@kernel inbounds=true function testneF!(
    mp      ::KernelParticle2D{T1, T2}, 
    pts_attr::KernelParticleProperty{T1, T2},
    ΔT      ::T2) where {T1, T2}
    # DV: dynamic viscosity
    FNUM_1 = T2(1); FNUM_12 = T2(0.5); FNUM_23 = T2(2/3); FNUM_1t = T2(1/ΔT); Cγ = T2(7)
    FNUM_2 = T2(2); FNUM_43 = T2(4/3); FNUM_13 = T2(1/3); DV      = T2(1e-3)
    ix = @index(Global)
    if ix ≤ mp.num
        pid = pts_attr.layer[ix]
        if pid == 1
            P = (pts_attr.Ks[pid])*((mp.ρs[ix]/mp.ρs_init[ix])^Cγ-FNUM_1)
            mp.σij[ix, 1] = mp.σm[ix]-P
            mp.σij[ix, 2] = mp.σm[ix]-P
            mp.σij[ix, 3] = mp.σm[ix]-P
            mp.σij[ix, 4] += FNUM_2*DV*mp.Δϵij_s[ix, 4]*FNUM_1t
            #mp.σm[ix] = P
        elseif pid == 2
            Ks   = pts_attr.Ks[pid]
            G    = pts_attr.G[pid]
            ωxy  = FNUM_12*(mp.∂Fs[ix, 2]-mp.∂Fs[ix, 3])
            σij1 = mp.σij[ix, 1]
            σij2 = mp.σij[ix, 2]
            σij4 = mp.σij[ix, 4]
            mp.σij[ix, 1] +=  ωxy*σij4*FNUM_2
            mp.σij[ix, 2] += -ωxy*σij4*FNUM_2
            mp.σij[ix, 4] +=  ωxy*(σij2-σij1)
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
end

function testprocedure!(args    ::MODELARGS, 
                        grid    ::GRID, 
                        mp      ::PARTICLE, 
                        pts_attr::PROPERTY,
                        bc      ::BOUNDARY,
                        ΔT      ::T2,
                        Ti      ::T2,
                                ::Val{:OS},
                                ::Val{:MUSL}) where {T2}
    Ti < args.Te ? G = args.gravity / args.Te * Ti : G = args.gravity
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
    G2P_OS!(dev)(ndrange=mp.num, grid, mp)
    testneF!(dev)(ndrange=mp.num, mp, pts_attr, ΔT)
    if args.MVL == true
        vollock1_OS!(dev)(ndrange=mp.num, grid, mp)
        vollock2_OS!(dev)(ndrange=mp.num, grid, mp)
    end
    return nothing
end

init_grid_space_x = 0.003
init_grid_space_y = 0.003
init_grid_range_x = [-0.1, 0.7]
init_grid_range_y = [-0.1, 0.6]
init_mp_in_space  = 2
init_project_name = "2d_water_structure"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :newtonfluid
init_gravity      = -9.8
init_ζs           = 0
# water 
init_ρs1          = 1e3
init_ν1           = 0
init_Ks1          = 5e6 # smaller than real value for smaller time step
init_E1           = 0
init_G1           = init_E1/(2*(1+init_ν1))
# solid (elastic obstacle)
init_ρs2          = 2700
init_ν2           = 0.4
init_E2           = 1e6
init_Ks2          = init_E2/(3*(1-2*init_ν2))
init_G2           = init_E2/(2*(1+  init_ν2))
init_T            = 1
init_Te           = 0
init_ΔT           = 0.05*min(init_grid_space_x, init_grid_space_y)/sqrt(init_E2/init_ρs2)
init_step         = floor(init_T/init_ΔT/100) |> Int64
init_step<10 ? init_step=1 : nothing
init_phase        = 1   
init_NIC          = 4
init_basis        = :linear
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
    animation    = true,
    hdf5         = true,
    hdf5_step    = init_step,
    MVL          = false,
    device       = :CUDA,
    coupling     = :OS,
    αT           = 0.05,
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
    phase    = init_phase,
    NIC      = init_NIC
)

# material points setup
range_x  = [0+grid.space_x/init_mp_in_space/2, 0.146-grid.space_x/init_mp_in_space/2]
range_y  = [0+grid.space_y/init_mp_in_space/2, 0.292-grid.space_y/init_mp_in_space/2]
space_x  = grid.space_x/init_mp_in_space
space_y  = grid.space_y/init_mp_in_space
num_x    = length(range_x[1]:space_x:range_x[2]) 
num_y    = length(range_y[1]:space_y:range_y[2])
x_tmp1   = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp1   = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
idxlayer = length(x_tmp1)
range_x  = [0.292+grid.space_x/init_mp_in_space/2, 0.304-grid.space_x/init_mp_in_space/2]
range_y  = [0.000+grid.space_y/init_mp_in_space/2, 0.080-grid.space_y/init_mp_in_space/2]
num_x    = length(range_x[1]:space_x:range_x[2])
num_y    = length(range_y[1]:space_y:range_y[2])
x_tmp2   = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp2   = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
x_tmp    = [x_tmp1; x_tmp2]
y_tmp    = [y_tmp1; y_tmp2]
mp_num   = length(x_tmp)
mp_layer = ones(mp_num)
mp_pos   = [x_tmp y_tmp]
mp_layer[idxlayer+1:end] .= 2
mp_ρs = ones(mp_num)
mp_ρs[1:idxlayer] .= init_ρs1
mp_ρs[idxlayer+1:end] .= init_ρs2
mp = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=mp_pos, ρs=mp_ρs,
    phase=init_phase, NIC=init_NIC)
ini_range = 1:idxlayer
mp.σm[ini_range] .= mp.ρs_init[ini_range].*(-9.8).*(
    (mp.pos[ini_range, 2].-maximum(mp.pos[ini_range, 2])).*(
    (mp.pos[ini_range, 1].-maximum(mp.pos[ini_range, 1]))./maximum(mp.pos[ini_range, 1]))
)
mp.σij[ini_range, 1:3] .= mp.σm[ini_range]

# particle property setup
mp_ν     = [init_ν1 , init_ν2 ]
mp_E     = [init_E1 , init_E2 ]
mp_G     = [init_G1 , init_G2 ]
mp_Ks    = [init_Ks1, init_Ks2]
pts_attr = ParticleProperty{iInt, iFloat}(layer=mp_layer, ν=mp_ν, E=mp_E, G=mp_G, Ks=mp_Ks)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
tmp_idx = findall(i->(grid.pos[i, 1]≤0)||(grid.pos[i, 1]≥0.584)||
                     ((0.291≤grid.pos[i, 1]≤0.304)&&(0≤grid.pos[i, 2]≤0.04)), 1:grid.node_num)
tmp_idy = findall(i->(grid.pos[i, 2]≤0)||grid.pos[i, 2]≥0.5, 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
)

# MPM solver
materialpointsolver!(args, grid, mp, pts_attr, bc, workflow=testprocedure!)

# post-processing
let 
    figfont    = MaterialPointSolver.fonttnr
    assets_dir = MaterialPointSolver.assets_dir
    idelsohn   = readdlm(joinpath(assets_dir, "data/2d_water_structure/idelsohn.csv"), ',')
    li         = readdlm(joinpath(assets_dir, "data/2d_water_structure/li.csv"      ), ',')
    rafiee     = readdlm(joinpath(assets_dir, "data/2d_water_structure/rafiee.csv"  ), ',')
    ryzhakov   = readdlm(joinpath(assets_dir, "data/2d_water_structure/ryzhakov.csv"), ',')
    walhorn    = readdlm(joinpath(assets_dir, "data/2d_water_structure/walhorn.csv" ), ',')
    mpmdata    = Vector{}[]
    hdf5_path  = joinpath(args.project_path, "$(args.project_name).h5")
    fid        = h5open(hdf5_path, "r")
    itr        = (read(fid["FILE_NUM"])-1) |> Int64
    mp_pos     = Observable(fid["mp_init"] |> read)
    mp_num     = length(mp_pos[][:, 1])
    xcol_idx   = findall(i->mp.init[i, 1]==mp.init[idxlayer+1, 1], 1:mp.num)
    mpindex    = findall(i->mp.init[i, 1]==mp.init[idxlayer+1, 1]&&
                            mp.init[i, 2]==mp.init[
                                xcol_idx[findmax(mp.init[xcol_idx, 2])[2]], 2
                            ], 1:mp.num)[1]
    for i in 1:itr
        cur_time = fid["group$(i)/time"] |> read
        if cur_time>0.1
            mp_pos = fid["group$(i)/mp_pos"] |> read
            push!(mpmdata, [cur_time, mp_pos[mpindex, 1]-mp.init[mpindex, 1]])
        end
    end
    close(fid)
    tmpx = [point[1] for point in mpmdata]
    tmpy = [point[2] for point in mpmdata]
    new_mpm = hcat(tmpx, tmpy)

    fig = Figure(size=(900, 700), fonts=(; regular=figfont, bold=figfont), fontsize=26)
    ax  = Axis(fig[1, 1], xlabel="Time [s]", ylabel="Deflection [m]")

    lines!(ax, idelsohn, linestyle=:dash      , linewidth=3, label="Idelsohn et al.")
    lines!(ax, li      , linestyle=:dot       , linewidth=3, label="Li et al."      )
    lines!(ax, rafiee  , linestyle=:dashdot   , linewidth=3, label="Rafiee et al."  )
    lines!(ax, ryzhakov, linestyle=:dashdotdot, linewidth=3, label="Ryzhakov et al.")
    lines!(ax, walhorn , linestyle= nothing   , linewidth=3, label="Walhorn et al." )
    scatterlines!(ax, new_mpm, linewidth=2, strokewidth=0, label="MPM", color=:red, 
        markersize=8)

    axislegend(ax, merge=true, unique=true, position=:rt, labelsize=20)
    limits!(ax, -0.05, 0.95, -0.025, 0.062)

    display(fig)
    save(joinpath(rtsdir, args.project_name, "$(args.project_name).png"), fig)
    @info "Figure saved in project path"
end