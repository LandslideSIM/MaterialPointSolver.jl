#==========================================================================================+
|                MPMSolver.jl: High-performance MPM Solver for Geomechanics                |
+------------------------------------------------------------------------------------------+
|  File Name  : 2d_slumping.jl                                                             |
|  Description: Please run this file in VSCode with Julia ENV                              |
|  Programmer : Zenan Huo                                                                  |
|  Start Date : 01/01/2022                                                                 |
|  Affiliation: Risk Group, UNIL-ISTE                                                      |
|  Test Case  : 2D slumping in Drucker-Prager constitutive model.                          |
+==========================================================================================#

init_grid_space_x = 0.1
init_grid_space_y = 0.1
init_grid_range_x = [-3, 113]
init_grid_range_y = [-3,  38]
init_mp_in_space  = 2
init_project_name = "2d_slumping"
init_project_path = joinpath(rtsdir, init_project_name)
init_constitutive = :druckerprager
init_gravity      = -9.8
init_ζ            = 0
init_ρs           = 2100
init_ν            = 0.3
init_E            = 70e6
init_G            = init_E/(2*(1+  init_ν))
init_Ks           = init_E/(3*(1-2*init_ν))
init_T            = 7
init_Te           = 0
init_ΔT           = 0.5*min(init_grid_space_x, init_grid_space_y)/sqrt(init_E/init_ρs)
init_step         = floor(init_T/init_ΔT/20) |> Int64
init_step<10 ? init_step=1 : nothing
init_σt           = 27.48e3
init_ϕ            = 20*π/180
init_c            = 1e4
init_ψ            = 0*π/180
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
    ζ            = init_ζ,
    gravity      = init_gravity,
    project_name = init_project_name,
    project_path = init_project_path,
    constitutive = init_constitutive,
    animation    = false,
    hdf5         = false,
    hdf5_step    = init_step,
    vollock      = true,
    device       = :CUDA,
    coupling     = :OS,
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
range_x    = [0+grid.space_x/init_mp_in_space/2, 110-grid.space_x/init_mp_in_space/2]
range_y    = [0+grid.space_y/init_mp_in_space/2,  35-grid.space_y/init_mp_in_space/2]
space_x    = grid.space_x/init_mp_in_space
space_y    = grid.space_y/init_mp_in_space
num_x      = length(range_x[1]:space_x:range_x[2])
num_y      = length(range_y[1]:space_y:range_y[2])
x_tmp      = repeat((range_x[1]:space_x:range_x[2])', num_y, 1) |> vec
y_tmp      = repeat((range_y[1]:space_y:range_y[2]) , 1, num_x) |> vec
a = findall(i->((30≤x_tmp[i]≤50)&&((65-x_tmp[i])≤y_tmp[i])) ||
               ((50≤x_tmp[i]   )&&( 15≤y_tmp[i])), 1:length(x_tmp))
map(i->splice!(i, a), [x_tmp, y_tmp])
mp_num   = length(x_tmp)
mp_ρs    = ones(mp_num).*init_ρs
mp_layer = ones(mp_num)
mp_pos   = [x_tmp y_tmp]
mp_ν     = [init_ν]
mp_E     = [init_E]
mp_G     = [init_G]
mp_σt    = [init_σt]
mp_ϕ     = [init_ϕ]
mp_c     = [init_c]
mp_ψ     = [init_ψ]
mp_Ks    = [init_Ks]
mp       = Particle2D{iInt, iFloat}(space_x=space_x, space_y=space_y, pos=mp_pos, ν=mp_ν, 
    E=mp_E, G=mp_G, σt=mp_σt, ϕ=mp_ϕ, c=mp_c, ψ=mp_ψ, ρs=mp_ρs, Ks=mp_Ks, layer=mp_layer, 
    phase=init_phase, NIC=init_NIC)

# boundary condition nodes index
vx_idx  = zeros(iInt, grid.node_num)
vy_idx  = zeros(iInt, grid.node_num) 
tmp_idx = findall(i->((grid.pos[i, 1]≤0)||(grid.pos[i, 1]≥110)||
                      (grid.pos[i, 2]≤0)), 1:grid.node_num)
tmp_idy = findall(i->( grid.pos[i, 2]≤0 ), 1:grid.node_num)
vx_idx[tmp_idx] .= 1
vy_idx[tmp_idy] .= 1
bc = VBoundary2D{iInt, iFloat}(
    Vx_s_Idx = vx_idx,
    Vx_s_Val = zeros(grid.node_num),
    Vy_s_Idx = vy_idx,
    Vy_s_Val = zeros(grid.node_num),
)

# MPM solver
# CUDA.@profile external=true simulate!(args, grid, mp, bc)
simulate!(args, grid, mp, bc)

# effective memory throughput
nio   = mp.num*124+mp.num*23*mp.NIC+grid.node_num*23+2*grid.cell_num
Aeff  = nio*sizeof(typeof(mp).parameters[2])*(1/1024^3)
MTeff = @sprintf "%.2e" (Aeff*args.iter_num)/(args.end_time-args.start_time)
@info "MTeff: $(MTeff) GB/s"


#= visualization
@views let
    figfontsize     = 66
    figc            = log10.(mp.epII)
    figmarkersize   = 2
    figmarkershape  = :circle
    figrangex       = [minimum(grid.pos[:, 1]), maximum(grid.pos[:, 1])]
    figrangey       = [minimum(grid.pos[:, 2]), maximum(grid.pos[:, 2])]
    figtickwidth    = 3
    figticklength   = 20
    figcbwidth      = 60
    figcbspinewidth = 0
    figcbticklength = 20
    figcbtickwidth  = 3
    figsize         = (1920, 780)
    figfont         = joinpath(assets, "fonts/tnr.ttf")
    figxlabel       = L"x\,(m)"
    figylabel       = L"y\,(m)"
    figcolormap     = :jet
    figtitle        = join([args.project_name, args.device])

    fig  = Figure(size=figsize, fontsize=figfontsize, font=figfont)
    axis = Axis(fig[1, 1], xlabel=figxlabel, ylabel=figylabel, title=figtitle,
        titlefont="Times New Roman", titlegap=12, spinewidth=3, aspect=DataAspect(),
        xticks=0:20:110, yticks=5:10:35, xticksize=figticklength, yticksize=figticklength,
        xtickwidth=figtickwidth, ytickwidth=figtickwidth, xticklabelfont=figfont,
        yticklabelfont=figfont)
    cb = scatter!(axis, mp.pos, color=figc, colormap=figcolormap, marker=figmarkershape,
        label="Particles", markersize=figmarkersize, strokewidth=0,
        colorrange=(-0.65, 0.65))
    limits!(axis, figrangex[1], figrangex[2], figrangey[1], figrangey[2])
    Colorbar(fig[1, 2], cb, label=L"log_{10}(\epsilon^{p})", size=figcbwidth,
        spinewidth=figcbspinewidth, ticksize=figcbticklength, tickwidth=figcbtickwidth,
        ticklabelfont=figfont)
    colsize!(fig.layout, 1, Aspect(1, 2.8))
    resize_to_layout!(fig)
    display(fig)
    save(joinpath(rtsdir, args.project_name, "$(figtitle).png"), fig)
    @info "Figure saved in project path"
end
=#