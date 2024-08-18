@kernel inbounds = true function fill_halo1!(
    halo_idx, 
    send_buff0_Ms, 
    send_buff0_Ps, 
    send_buff0_Fs, 
    grid
)
    ix = @index(Global)
    if ix ≤ length(halo_idx)
        send_buff0_Ms[ix]    = grid.Ms[halo_idx[ix]]
        send_buff0_Ps[ix, 1] = grid.Ps[halo_idx[ix], 1]
        send_buff0_Ps[ix, 2] = grid.Ps[halo_idx[ix], 2]
        send_buff0_Ps[ix, 3] = grid.Ps[halo_idx[ix], 3]
        send_buff0_Fs[ix, 1] = grid.Fs[halo_idx[ix], 1]
        send_buff0_Fs[ix, 2] = grid.Fs[halo_idx[ix], 2]
        send_buff0_Fs[ix, 3] = grid.Fs[halo_idx[ix], 3]
    end
end

@kernel inbounds = true function fill_halo2!(
    halo_idx, 
    send_buff0_Ps, 
    grid
)
    ix = @index(Global)
    if ix ≤ length(halo_idx)
        send_buff0_Ps[ix, 1] = grid.Ps[halo_idx[ix], 1]
        send_buff0_Ps[ix, 2] = grid.Ps[halo_idx[ix], 2]
        send_buff0_Ps[ix, 3] = grid.Ps[halo_idx[ix], 3]
    end
end

@kernel inbounds = true function update_halo1!(
    halo_idx, 
    recv_buff0_Ms, 
    recv_buff0_Ps, 
    recv_buff0_Fs, 
    grid
)
    ix = @index(Global)
    if ix <= length(halo_idx)
        grid.Ms[halo_idx[ix]]    += recv_buff0_Ms[ix]
        grid.Ps[halo_idx[ix], 1] += recv_buff0_Ps[ix, 1]
        grid.Ps[halo_idx[ix], 2] += recv_buff0_Ps[ix, 2]
        grid.Ps[halo_idx[ix], 3] += recv_buff0_Ps[ix, 3]
        grid.Fs[halo_idx[ix], 1] += recv_buff0_Fs[ix, 1]
        grid.Fs[halo_idx[ix], 2] += recv_buff0_Fs[ix, 2]
        grid.Fs[halo_idx[ix], 3] += recv_buff0_Fs[ix, 3]
        recv_buff0_Ms[ix]    = 0f0
        recv_buff0_Ps[ix, 1] = 0f0
        recv_buff0_Ps[ix, 2] = 0f0
        recv_buff0_Ps[ix, 3] = 0f0
        recv_buff0_Fs[ix, 1] = 0f0
        recv_buff0_Fs[ix, 2] = 0f0
        recv_buff0_Fs[ix, 3] = 0f0
    end
end

@kernel inbounds = true function update_halo2!(
    halo_idx, 
    recv_buff0_Ps, 
    grid
)
    ix = @index(Global)
    if ix <= length(halo_idx)
        grid.Ps[halo_idx[ix], 1] += recv_buff0_Ps[ix, 1]
        grid.Ps[halo_idx[ix], 2] += recv_buff0_Ps[ix, 2]
        grid.Ps[halo_idx[ix], 3] += recv_buff0_Ps[ix, 3]
        recv_buff0_Ps[ix, 1] = 0f0
        recv_buff0_Ps[ix, 2] = 0f0
        recv_buff0_Ps[ix, 3] = 0f0
    end
end