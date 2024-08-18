
abstract type     GPUGRID end
abstract type GPUPARTICLE end
abstract type      GPUVBC end

struct GPUGrid2D{T1<:Signed, T2<:AbstractFloat} <: GPUGRID
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
    pos       ::CuArray{T2, 2}
    cell_num_x::T1
    cell_num_y::T1
    cell_num  ::T1
    c2n       ::CuArray{T1, 2}
    σm        ::CuArray{T2, 1}
    σw        ::CuArray{T2, 1}
    vol       ::CuArray{T2, 1}
    Ms        ::CuArray{T2, 1}
    Mw        ::CuArray{T2, 1}
    Mi        ::CuArray{T2, 1}
    Ps        ::CuArray{T2, 2}
    Pw        ::CuArray{T2, 2}
    Vs        ::CuArray{T2, 2}
    Vw        ::CuArray{T2, 2}
    Vs_T      ::CuArray{T2, 2}
    Vw_T      ::CuArray{T2, 2}
    Fs        ::CuArray{T2, 2}
    Fw        ::CuArray{T2, 2}
    Fdrag     ::CuArray{T2, 2}
    a_s       ::CuArray{T2, 2}
    a_w       ::CuArray{T2, 2}
    Δd_s      ::CuArray{T2, 2}
    Δd_w      ::CuArray{T2, 2}
end

struct KernelGrid2D{T1<:Signed, T2<:AbstractFloat} <: GPUGRID
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
    pos       ::CuDeviceMatrix{T2, 1}
    cell_num_x::T1
    cell_num_y::T1
    cell_num  ::T1
    c2n       ::CuDeviceMatrix{T1, 1}
    σm        ::CuDeviceVector{T2, 1}
    σw        ::CuDeviceVector{T2, 1}
    vol       ::CuDeviceVector{T2, 1}
    Ms        ::CuDeviceVector{T2, 1}
    Mw        ::CuDeviceVector{T2, 1}
    Mi        ::CuDeviceVector{T2, 1}
    Ps        ::CuDeviceMatrix{T2, 1}
    Pw        ::CuDeviceMatrix{T2, 1}
    Vs        ::CuDeviceMatrix{T2, 1}
    Vw        ::CuDeviceMatrix{T2, 1}
    Vs_T      ::CuDeviceMatrix{T2, 1}
    Vw_T      ::CuDeviceMatrix{T2, 1}
    Fs        ::CuDeviceMatrix{T2, 1}
    Fw        ::CuDeviceMatrix{T2, 1}
    Fdrag     ::CuDeviceMatrix{T2, 1}
    a_s       ::CuDeviceMatrix{T2, 1}
    a_w       ::CuDeviceMatrix{T2, 1}
    Δd_s      ::CuDeviceMatrix{T2, 1}
    Δd_w      ::CuDeviceMatrix{T2, 1}
end

Adapt.adapt_structure(to, s::GPUGrid2D) = KernelGrid2D(
    adapt(to, s.range_x1  ),
    adapt(to, s.range_x2  ),
    adapt(to, s.range_y1  ),
    adapt(to, s.range_y2  ),
    adapt(to, s.space_x   ),
    adapt(to, s.space_y   ),
    adapt(to, s.phase     ),
    adapt(to, s.node_num_x),
    adapt(to, s.node_num_y),
    adapt(to, s.node_num  ),
    adapt(to, s.NIC       ),
    adapt(to, s.pos       ),
    adapt(to, s.cell_num_x),
    adapt(to, s.cell_num_y),
    adapt(to, s.cell_num  ),
    adapt(to, s.c2n       ),
    adapt(to, s.σm        ),
    adapt(to, s.σw        ),
    adapt(to, s.vol       ),
    adapt(to, s.Ms        ),
    adapt(to, s.Mw        ),
    adapt(to, s.Mi        ),
    adapt(to, s.Ps        ),
    adapt(to, s.Pw        ),
    adapt(to, s.Vs        ),
    adapt(to, s.Vw        ),
    adapt(to, s.Vs_T      ),
    adapt(to, s.Vw_T      ),
    adapt(to, s.Fs        ),
    adapt(to, s.Fw        ),
    adapt(to, s.Fdrag     ),
    adapt(to, s.a_s       ),
    adapt(to, s.a_w       ),
    adapt(to, s.Δd_s      ),
    adapt(to, s.Δd_w      )
)

"""
    struct GPUGrid2D{T1<:Signed, T2<:AbstractFloat}

Description:
---
Grid2D GPU struct. See Grid2D for more details.
"""
@kwdef struct GPUGrid3D{T1<:Signed, T2<:AbstractFloat} <: GPUGRID
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
    pos       ::CuArray{T2, 2}
    cell_num_x::T1
    cell_num_y::T1
    cell_num_z::T1
    cell_num  ::T1
    c2n       ::CuArray{T1, 2}
    σm        ::CuArray{T2, 1}
    σw        ::CuArray{T2, 1}
    vol       ::CuArray{T2, 1}
    Ms        ::CuArray{T2, 1}
    Mw        ::CuArray{T2, 1}
    Mi        ::CuArray{T2, 1}
    Ps        ::CuArray{T2, 2}
    Pw        ::CuArray{T2, 2}
    Vs        ::CuArray{T2, 2}
    Vw        ::CuArray{T2, 2}
    Vs_T      ::CuArray{T2, 2}
    Vw_T      ::CuArray{T2, 2}
    Fs        ::CuArray{T2, 2}
    Fw        ::CuArray{T2, 2}
    Fdrag     ::CuArray{T2, 2}
    a_s       ::CuArray{T2, 2}
    a_w       ::CuArray{T2, 2}
    Δd_s      ::CuArray{T2, 2}
    Δd_w      ::CuArray{T2, 2}
end

struct KernelGrid3D{T1<:Signed, T2<:AbstractFloat} <: GPUGRID
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
    pos       ::CuDeviceMatrix{T2, 1}
    cell_num_x::T1
    cell_num_y::T1
    cell_num_z::T1
    cell_num  ::T1
    c2n       ::CuDeviceMatrix{T1, 1}
    σm        ::CuDeviceVector{T2, 1}
    σw        ::CuDeviceVector{T2, 1}
    vol       ::CuDeviceVector{T2, 1}
    Ms        ::CuDeviceVector{T2, 1}
    Mw        ::CuDeviceVector{T2, 1}
    Mi        ::CuDeviceVector{T2, 1}
    Ps        ::CuDeviceMatrix{T2, 1}
    Pw        ::CuDeviceMatrix{T2, 1}
    Vs        ::CuDeviceMatrix{T2, 1}
    Vw        ::CuDeviceMatrix{T2, 1}
    Vs_T      ::CuDeviceMatrix{T2, 1}
    Vw_T      ::CuDeviceMatrix{T2, 1}
    Fs        ::CuDeviceMatrix{T2, 1}
    Fw        ::CuDeviceMatrix{T2, 1}
    Fdrag     ::CuDeviceMatrix{T2, 1}
    a_s       ::CuDeviceMatrix{T2, 1}
    a_w       ::CuDeviceMatrix{T2, 1}
    Δd_s      ::CuDeviceMatrix{T2, 1}
    Δd_w      ::CuDeviceMatrix{T2, 1}
end

Adapt.adapt_structure(to, s::GPUGrid3D) = KernelGrid3D(
    adapt(to, s.range_x1  ),
    adapt(to, s.range_x2  ),
    adapt(to, s.range_y1  ),
    adapt(to, s.range_y2  ),
    adapt(to, s.range_z1  ),
    adapt(to, s.range_z2  ),
    adapt(to, s.space_x   ),
    adapt(to, s.space_y   ),
    adapt(to, s.space_z   ),
    adapt(to, s.phase     ),
    adapt(to, s.node_num_x),
    adapt(to, s.node_num_y),
    adapt(to, s.node_num_z),
    adapt(to, s.node_num  ),
    adapt(to, s.NIC       ),
    adapt(to, s.pos       ),
    adapt(to, s.cell_num_x),
    adapt(to, s.cell_num_y),
    adapt(to, s.cell_num_z),
    adapt(to, s.cell_num  ),
    adapt(to, s.c2n       ),
    adapt(to, s.σm        ),
    adapt(to, s.σw        ),
    adapt(to, s.vol       ),
    adapt(to, s.Ms        ),
    adapt(to, s.Mw        ),
    adapt(to, s.Mi        ),
    adapt(to, s.Ps        ),
    adapt(to, s.Pw        ),
    adapt(to, s.Vs        ),
    adapt(to, s.Vw        ),
    adapt(to, s.Vs_T      ),
    adapt(to, s.Vw_T      ),
    adapt(to, s.Fs        ),
    adapt(to, s.Fw        ),
    adapt(to, s.Fdrag     ),
    adapt(to, s.a_s       ),
    adapt(to, s.a_w       ),
    adapt(to, s.Δd_s      ),
    adapt(to, s.Δd_w      )
)

"""
    struct GPUParticle2D{T1<:Signed, T2<:AbstractFloat}

Description:
---
Particle2D GPU struct. See Particle2D for more details.
"""
@kwdef mutable struct GPUParticle2D{T1<:Signed, T2<:AbstractFloat} <: GPUPARTICLE
    num     ::T1
    phase   ::T1
    NIC     ::T1
    space_x ::T2
    space_y ::T2
    p2c     ::CuArray{T1, 1}
    p2n     ::CuArray{T1, 2}
    pos     ::CuArray{T2, 2}
    σm      ::CuArray{T2, 1}
    J       ::CuArray{T2, 1}
    epII    ::CuArray{T2, 1}
    epK     ::CuArray{T2, 1}
    vol     ::CuArray{T2, 1}
    vol_init::CuArray{T2, 1}
    Ms      ::CuArray{T2, 1}
    Mw      ::CuArray{T2, 1}
    Mi      ::CuArray{T2, 1}
    porosity::CuArray{T2, 1}
    cfl     ::CuArray{T2, 1}
    k       ::CuArray{T2, 1}
    ρs      ::CuArray{T2, 1}
    ρs_init ::CuArray{T2, 1}
    ρw      ::CuArray{T2, 1}
    ρw_init ::CuArray{T2, 1}
    ν       ::CuArray{T2, 1}
    E       ::CuArray{T2, 1}
    G       ::CuArray{T2, 1}
    Ks      ::CuArray{T2, 1}
    Kw      ::CuArray{T2, 1}
    σt      ::CuArray{T2, 1}
    ϕ       ::CuArray{T2, 1}
    c       ::CuArray{T2, 1}
    cr      ::CuArray{T2, 1}
    Hp      ::CuArray{T2, 1}
    ψ       ::CuArray{T2, 1}    
    init    ::CuArray{T2, 2}
    γ       ::CuArray{T2, 1}
    B       ::CuArray{T2, 1}
    σw      ::CuArray{T2, 1}
    σij     ::CuArray{T2, 2}
    ϵij_s   ::CuArray{T2, 2}
    ϵij_w   ::CuArray{T2, 2}
    Δϵij_s  ::CuArray{T2, 2}
    Δϵij_w  ::CuArray{T2, 2}
    sij     ::CuArray{T2, 2}
    Vs      ::CuArray{T2, 2}
    Vw      ::CuArray{T2, 2}
    Ps      ::CuArray{T2, 2}
    Pw      ::CuArray{T2, 2}
    Ni      ::CuArray{T2, 2}
    ∂Nx     ::CuArray{T2, 2}
    ∂Ny     ::CuArray{T2, 2}
    ∂Fs     ::CuArray{T2, 2}
    ∂Fw     ::CuArray{T2, 2}
    F       ::CuArray{T2, 2}
    layer   ::CuArray{T1, 1}
end

struct KernelParticle2D{T1<:Signed, T2<:AbstractFloat} <: GPUPARTICLE
    num     ::T1
    phase   ::T1
    NIC     ::T1
    space_x ::T2
    space_y ::T2
    p2c     ::CuDeviceVector{T1, 1}
    p2n     ::CuDeviceMatrix{T1, 1}
    pos     ::CuDeviceMatrix{T2, 1}
    σm      ::CuDeviceVector{T2, 1}
    J       ::CuDeviceVector{T2, 1}
    epII    ::CuDeviceVector{T2, 1}
    epK     ::CuDeviceVector{T2, 1}
    vol     ::CuDeviceVector{T2, 1}
    vol_init::CuDeviceVector{T2, 1}
    Ms      ::CuDeviceVector{T2, 1}
    Mw      ::CuDeviceVector{T2, 1}
    Mi      ::CuDeviceVector{T2, 1}
    porosity::CuDeviceVector{T2, 1}
    cfl     ::CuDeviceVector{T2, 1}
    k       ::CuDeviceVector{T2, 1}
    ρs      ::CuDeviceVector{T2, 1}
    ρs_init ::CuDeviceVector{T2, 1}
    ρw      ::CuDeviceVector{T2, 1}
    ρw_init ::CuDeviceVector{T2, 1}
    ν       ::CuDeviceVector{T2, 1}
    E       ::CuDeviceVector{T2, 1}
    G       ::CuDeviceVector{T2, 1}
    Ks      ::CuDeviceVector{T2, 1}
    Kw      ::CuDeviceVector{T2, 1}
    σt      ::CuDeviceVector{T2, 1}
    ϕ       ::CuDeviceVector{T2, 1}
    c       ::CuDeviceVector{T2, 1}
    cr      ::CuDeviceVector{T2, 1}
    Hp      ::CuDeviceVector{T2, 1}
    ψ       ::CuDeviceVector{T2, 1}    
    init    ::CuDeviceMatrix{T2, 1}
    γ       ::CuDeviceVector{T2, 1}
    B       ::CuDeviceVector{T2, 1}
    σw      ::CuDeviceVector{T2, 1}
    σij     ::CuDeviceMatrix{T2, 1}
    ϵij_s   ::CuDeviceMatrix{T2, 1}
    ϵij_w   ::CuDeviceMatrix{T2, 1}
    Δϵij_s  ::CuDeviceMatrix{T2, 1}
    Δϵij_w  ::CuDeviceMatrix{T2, 1}
    sij     ::CuDeviceMatrix{T2, 1}
    Vs      ::CuDeviceMatrix{T2, 1}
    Vw      ::CuDeviceMatrix{T2, 1}
    Ps      ::CuDeviceMatrix{T2, 1}
    Pw      ::CuDeviceMatrix{T2, 1}
    Ni      ::CuDeviceMatrix{T2, 1}
    ∂Nx     ::CuDeviceMatrix{T2, 1}
    ∂Ny     ::CuDeviceMatrix{T2, 1}
    ∂Fs     ::CuDeviceMatrix{T2, 1}
    ∂Fw     ::CuDeviceMatrix{T2, 1}
    F       ::CuDeviceMatrix{T2, 1}
    layer   ::CuDeviceVector{T1, 1}
end

Adapt.adapt_structure(to, s::GPUParticle2D) = KernelParticle2D(
    adapt(to, s.num     ),
    adapt(to, s.phase   ),
    adapt(to, s.NIC     ),
    adapt(to, s.space_x ),
    adapt(to, s.space_y ),
    adapt(to, s.p2c     ),
    adapt(to, s.p2n     ),
    adapt(to, s.pos     ),
    adapt(to, s.σm      ),
    adapt(to, s.J       ),
    adapt(to, s.epII    ),
    adapt(to, s.epK     ),
    adapt(to, s.vol     ),
    adapt(to, s.vol_init),
    adapt(to, s.Ms      ),
    adapt(to, s.Mw      ),
    adapt(to, s.Mi      ),
    adapt(to, s.porosity),
    adapt(to, s.cfl     ),
    adapt(to, s.k       ),
    adapt(to, s.ρs      ),
    adapt(to, s.ρs_init ),
    adapt(to, s.ρw      ),
    adapt(to, s.ρw_init ),
    adapt(to, s.ν       ),
    adapt(to, s.E       ),
    adapt(to, s.G       ),
    adapt(to, s.Ks      ),
    adapt(to, s.Kw      ),
    adapt(to, s.σt      ),
    adapt(to, s.ϕ       ),
    adapt(to, s.c       ),
    adapt(to, s.cr      ),
    adapt(to, s.Hp      ),
    adapt(to, s.ψ       ),    
    adapt(to, s.init    ),
    adapt(to, s.γ       ),
    adapt(to, s.B       ),
    adapt(to, s.σw      ),
    adapt(to, s.σij     ),
    adapt(to, s.ϵij_s   ),
    adapt(to, s.ϵij_w   ),
    adapt(to, s.Δϵij_s  ),
    adapt(to, s.Δϵij_w  ),
    adapt(to, s.sij     ),
    adapt(to, s.Vs      ),
    adapt(to, s.Vw      ),
    adapt(to, s.Ps      ),
    adapt(to, s.Pw      ),
    adapt(to, s.Ni      ),
    adapt(to, s.∂Nx     ),
    adapt(to, s.∂Ny     ),
    adapt(to, s.∂Fs     ),
    adapt(to, s.∂Fw     ),
    adapt(to, s.F       ),
    adapt(to, s.layer   )
)

"""
    struct GPUParticle3D{T1<:Signed, T2<:AbstractFloat}

Description:
---
Particle3D GPU struct. See Particle3D for more details.
"""
@kwdef mutable struct GPUParticle3D{T1<:Signed, T2<:AbstractFloat} <: GPUPARTICLE
    num     ::T1
    phase   ::T1
    NIC     ::T1
    space_x ::T2
    space_y ::T2
    space_z ::T2
    p2c     ::CuArray{T1, 1}
    p2n     ::CuArray{T1, 2}
    pos     ::CuArray{T2, 2}
    σm      ::CuArray{T2, 1}
    J       ::CuArray{T2, 1}
    epII    ::CuArray{T2, 1}
    epK     ::CuArray{T2, 1}
    vol     ::CuArray{T2, 1}
    vol_init::CuArray{T2, 1}
    Ms      ::CuArray{T2, 1}
    Mw      ::CuArray{T2, 1}
    Mi      ::CuArray{T2, 1}
    porosity::CuArray{T2, 1}
    cfl     ::CuArray{T2, 1}
    k       ::CuArray{T2, 1}
    ρs      ::CuArray{T2, 1}
    ρs_init ::CuArray{T2, 1}
    ρw      ::CuArray{T2, 1}
    ρw_init ::CuArray{T2, 1}
    ν       ::CuArray{T2, 1}
    E       ::CuArray{T2, 1}
    G       ::CuArray{T2, 1}
    Ks      ::CuArray{T2, 1}
    Kw      ::CuArray{T2, 1}
    σt      ::CuArray{T2, 1}
    ϕ       ::CuArray{T2, 1}
    c       ::CuArray{T2, 1}
    cr      ::CuArray{T2, 1}
    Hp      ::CuArray{T2, 1}
    ψ       ::CuArray{T2, 1}
    init    ::CuArray{T2, 2}
    γ       ::CuArray{T2, 1}
    B       ::CuArray{T2, 1}
    σw      ::CuArray{T2, 1}
    σij     ::CuArray{T2, 2}
    ϵij_s   ::CuArray{T2, 2}
    ϵij_w   ::CuArray{T2, 2}
    Δϵij_s  ::CuArray{T2, 2}
    Δϵij_w  ::CuArray{T2, 2}
    sij     ::CuArray{T2, 2}
    Vs      ::CuArray{T2, 2}
    Vw      ::CuArray{T2, 2}
    Ps      ::CuArray{T2, 2}
    Pw      ::CuArray{T2, 2}
    Ni      ::CuArray{T2, 2}
    ∂Nx     ::CuArray{T2, 2}
    ∂Ny     ::CuArray{T2, 2}
    ∂Nz     ::CuArray{T2, 2}
    ∂Fs     ::CuArray{T2, 2}
    ∂Fw     ::CuArray{T2, 2}
    F       ::CuArray{T2, 2}
    layer   ::CuArray{T1, 1}
end

struct KernelParticle3D{T1<:Signed, T2<:AbstractFloat} <: GPUPARTICLE
    num     ::T1
    phase   ::T1
    NIC     ::T1
    space_x ::T2
    space_y ::T2
    space_z ::T2
    p2c     ::CuDeviceVector{T1, 1}
    p2n     ::CuDeviceMatrix{T1, 1}
    pos     ::CuDeviceMatrix{T2, 1}
    σm      ::CuDeviceVector{T2, 1}
    J       ::CuDeviceVector{T2, 1}
    epII    ::CuDeviceVector{T2, 1}
    epK     ::CuDeviceVector{T2, 1}
    vol     ::CuDeviceVector{T2, 1}
    vol_init::CuDeviceVector{T2, 1}
    Ms      ::CuDeviceVector{T2, 1}
    Mw      ::CuDeviceVector{T2, 1}
    Mi      ::CuDeviceVector{T2, 1}
    porosity::CuDeviceVector{T2, 1}
    cfl     ::CuDeviceVector{T2, 1}
    k       ::CuDeviceVector{T2, 1}
    ρs      ::CuDeviceVector{T2, 1}
    ρs_init ::CuDeviceVector{T2, 1}
    ρw      ::CuDeviceVector{T2, 1}
    ρw_init ::CuDeviceVector{T2, 1}
    ν       ::CuDeviceVector{T2, 1}
    E       ::CuDeviceVector{T2, 1}
    G       ::CuDeviceVector{T2, 1}
    Ks      ::CuDeviceVector{T2, 1}
    Kw      ::CuDeviceVector{T2, 1}
    σt      ::CuDeviceVector{T2, 1}
    ϕ       ::CuDeviceVector{T2, 1}
    c       ::CuDeviceVector{T2, 1}
    cr      ::CuDeviceVector{T2, 1}
    Hp      ::CuDeviceVector{T2, 1}
    ψ       ::CuDeviceVector{T2, 1}
    init    ::CuDeviceMatrix{T2, 1}
    γ       ::CuDeviceVector{T2, 1}
    B       ::CuDeviceVector{T2, 1}
    σw      ::CuDeviceVector{T2, 1}
    σij     ::CuDeviceMatrix{T2, 1}
    ϵij_s   ::CuDeviceMatrix{T2, 1}
    ϵij_w   ::CuDeviceMatrix{T2, 1}
    Δϵij_s  ::CuDeviceMatrix{T2, 1}
    Δϵij_w  ::CuDeviceMatrix{T2, 1}
    sij     ::CuDeviceMatrix{T2, 1}
    Vs      ::CuDeviceMatrix{T2, 1}
    Vw      ::CuDeviceMatrix{T2, 1}
    Ps      ::CuDeviceMatrix{T2, 1}
    Pw      ::CuDeviceMatrix{T2, 1}
    Ni      ::CuDeviceMatrix{T2, 1}
    ∂Nx     ::CuDeviceMatrix{T2, 1}
    ∂Ny     ::CuDeviceMatrix{T2, 1}
    ∂Nz     ::CuDeviceMatrix{T2, 1}
    ∂Fs     ::CuDeviceMatrix{T2, 1}
    ∂Fw     ::CuDeviceMatrix{T2, 1}
    F       ::CuDeviceMatrix{T2, 1}
    layer   ::CuDeviceVector{T1, 1}
end

Adapt.adapt_structure(to, s::GPUParticle3D) = KernelParticle3D(
    adapt(to, s.num     ),
    adapt(to, s.phase   ),
    adapt(to, s.NIC     ),
    adapt(to, s.space_x ),
    adapt(to, s.space_y ),
    adapt(to, s.space_z ),
    adapt(to, s.p2c     ),
    adapt(to, s.p2n     ),
    adapt(to, s.pos     ),
    adapt(to, s.σm      ),
    adapt(to, s.J       ),
    adapt(to, s.epII    ),
    adapt(to, s.epK     ),
    adapt(to, s.vol     ),
    adapt(to, s.vol_init),
    adapt(to, s.Ms      ),
    adapt(to, s.Mw      ),
    adapt(to, s.Mi      ),
    adapt(to, s.porosity),
    adapt(to, s.cfl     ),
    adapt(to, s.k       ),
    adapt(to, s.ρs      ),
    adapt(to, s.ρs_init ),
    adapt(to, s.ρw      ),
    adapt(to, s.ρw_init ),
    adapt(to, s.ν       ),
    adapt(to, s.E       ),
    adapt(to, s.G       ),
    adapt(to, s.Ks      ),
    adapt(to, s.Kw      ),
    adapt(to, s.σt      ),
    adapt(to, s.ϕ       ),
    adapt(to, s.c       ),
    adapt(to, s.cr      ),
    adapt(to, s.Hp      ),
    adapt(to, s.ψ       ),
    adapt(to, s.init    ),
    adapt(to, s.γ       ),
    adapt(to, s.B       ),
    adapt(to, s.σw      ),
    adapt(to, s.σij     ),
    adapt(to, s.ϵij_s   ),
    adapt(to, s.ϵij_w   ),
    adapt(to, s.Δϵij_s  ),
    adapt(to, s.Δϵij_w  ),
    adapt(to, s.sij     ),
    adapt(to, s.Vs      ),
    adapt(to, s.Vw      ),
    adapt(to, s.Ps      ),
    adapt(to, s.Pw      ),
    adapt(to, s.Ni      ),
    adapt(to, s.∂Nx     ),
    adapt(to, s.∂Ny     ),
    adapt(to, s.∂Nz     ),
    adapt(to, s.∂Fs     ),
    adapt(to, s.∂Fw     ),
    adapt(to, s.F       ),
    adapt(to, s.layer   )
)

@kwdef struct GPUVBoundary2D{T1<:Signed, T2<:AbstractFloat} <: GPUVBC
    Vx_s_Idx::CuArray{T1, 1}
    Vx_s_Val::CuArray{T2, 1}
    Vy_s_Idx::CuArray{T1, 1}
    Vy_s_Val::CuArray{T2, 1}
    Vx_w_Idx::CuArray{T1, 1}
    Vx_w_Val::CuArray{T2, 1}
    Vy_w_Idx::CuArray{T1, 1}
    Vy_w_Val::CuArray{T2, 1}
end

struct KernelVBoundary2D{T1<:Signed, T2<:AbstractFloat} <: GPUVBC
    Vx_s_Idx::CuDeviceVector{T1, 1}
    Vx_s_Val::CuDeviceVector{T2, 1}
    Vy_s_Idx::CuDeviceVector{T1, 1}
    Vy_s_Val::CuDeviceVector{T2, 1}
    Vx_w_Idx::CuDeviceVector{T1, 1}
    Vx_w_Val::CuDeviceVector{T2, 1}
    Vy_w_Idx::CuDeviceVector{T1, 1}
    Vy_w_Val::CuDeviceVector{T2, 1}
end

Adapt.adapt_structure(to, s::GPUVBoundary2D) = KernelVBoundary2D(
    adapt(to, s.Vx_s_Idx),
    adapt(to, s.Vx_s_Val),
    adapt(to, s.Vy_s_Idx),
    adapt(to, s.Vy_s_Val),
    adapt(to, s.Vx_w_Idx),
    adapt(to, s.Vx_w_Val),
    adapt(to, s.Vy_w_Idx),
    adapt(to, s.Vy_w_Val)
)

@kwdef struct GPUVBoundary3D{T1<:Signed, T2<:AbstractFloat} <: GPUVBC
    Vx_s_Idx::CuArray{T1, 1}
    Vx_s_Val::CuArray{T2, 1}
    Vy_s_Idx::CuArray{T1, 1}
    Vy_s_Val::CuArray{T2, 1}
    Vz_s_Idx::CuArray{T1, 1}
    Vz_s_Val::CuArray{T2, 1}
    Vx_w_Idx::CuArray{T1, 1}
    Vx_w_Val::CuArray{T2, 1}
    Vy_w_Idx::CuArray{T1, 1}
    Vy_w_Val::CuArray{T2, 1}
    Vz_w_Idx::CuArray{T1, 1}
    Vz_w_Val::CuArray{T2, 1}
end

struct KernelVBoundary3D{T1<:Signed, T2<:AbstractFloat} <: GPUVBC
    Vx_s_Idx::CuDeviceVector{T1, 1}
    Vx_s_Val::CuDeviceVector{T2, 1}
    Vy_s_Idx::CuDeviceVector{T1, 1}
    Vy_s_Val::CuDeviceVector{T2, 1}
    Vz_s_Idx::CuDeviceVector{T1, 1}
    Vz_s_Val::CuDeviceVector{T2, 1}
    Vx_w_Idx::CuDeviceVector{T1, 1}
    Vx_w_Val::CuDeviceVector{T2, 1}
    Vy_w_Idx::CuDeviceVector{T1, 1}
    Vy_w_Val::CuDeviceVector{T2, 1}
    Vz_w_Idx::CuDeviceVector{T1, 1}
    Vz_w_Val::CuDeviceVector{T2, 1}
end

Adapt.adapt_structure(to, s::GPUVBoundary3D) = KernelVBoundary3D(
    adapt(to, s.Vx_s_Idx),
    adapt(to, s.Vx_s_Val),
    adapt(to, s.Vy_s_Idx),
    adapt(to, s.Vy_s_Val),
    adapt(to, s.Vz_s_Idx),
    adapt(to, s.Vz_s_Val),
    adapt(to, s.Vx_w_Idx),
    adapt(to, s.Vx_w_Val),
    adapt(to, s.Vy_w_Idx),
    adapt(to, s.Vy_w_Val),
    adapt(to, s.Vz_w_Idx),
    adapt(to, s.Vz_w_Val)
)