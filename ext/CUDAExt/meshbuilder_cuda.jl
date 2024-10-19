function getparticle(geo_path::String, size_min, size_max, lp, ::Val{:CUDA})
    local node, tet
    @suppress node, tet = gmsh_mesh3D(geo_path, size_min, size_max)
    # terminal info
    @info """Gmsh results
    number of nodes     : $(size(node, 1))
    number of tetrahedra: $(size(tet, 1))
    """
    # get bounding box for particles
    min_x, max_x = minimum(node[:, 1]), maximum(node[:, 1])
    min_y, max_y = minimum(node[:, 2]), maximum(node[:, 2])
    min_z, max_z = minimum(node[:, 3]), maximum(node[:, 3])
    offsetx = (max_x - min_x) * 0.2
    offsety = (max_y - min_y) * 0.2
    offsetz = (max_z - min_z) * 0.2
    min_x -= offsetx
    max_x += offsetx
    min_y -= offsety
    max_y += offsety
    min_z -= offsetz
    max_z += offsetz
    # generate structured particles
    x_, y_, z_ = meshbuilder(min_x:lp:max_x, min_y:lp:max_y, min_z:lp:max_z)
    pts = hcat(x_, y_, z_)
    pts_num = size(pts, 1)
    results = Vector{Bool}(zeros(pts_num))
    # print terminal info
    datasize = Base.summarysize(results) + Base.summarysize(pts) +
    Base.summarysize(node   ) + Base.summarysize(tet)
    outprint = @sprintf("%.1f", datasize / 1024 ^ 3)
    dev_id   = CUDA.device().handle
    content  = "uploading [≈ $(outprint) GiB] → :CUDA [$(dev_id)]"
    println("\e[1;32m[▲ I/O:\e[0m \e[0;32m$(content)\e[0m")
    # upload data to device
    rst_dev = CuArray(results)
    pts_dev = CuArray(pts)
    nde_dev = CuArray(node)
    tet_dev = CuArray(tet)
    # run kernel
    pts_in_polyhedron!(CUDABackend())(ndrange=pts_num, pts_dev, nde_dev, tet_dev, rst_dev)
    copyto!(results, rst_dev) # download data from device
    # clean device
    CUDA.unsafe_free!(rst_dev)
    CUDA.unsafe_free!(pts_dev)
    CUDA.unsafe_free!(nde_dev)
    CUDA.unsafe_free!(tet_dev)
    CUDA.reclaim()
    # return pts in polyhedron
    return copy(pts[findall(i -> results[i], 1:pts_num), :])
end