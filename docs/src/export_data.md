# Vars saved in HDF5 (.h5)

!!! Note: 
- `Nmp` is the number of particle in the model
- `Nn` is the number of grid node in the model

## 1. Data saved in each iteration

| Var name |          Meaning          |   2D    |   3D    |
| :------: | :-----------------------: | :-----: | :-----: |
|   sig    |       total stress        | Nmp x 4 | Nmp x 6 |
|  eps_s   |       solid strain        | Nmp x 4 | Nmp x 6 |
|   epII   | equivlent plastic strain  | Nmp x 1 | Nmp x 1 |
|   epK    | volumetric plastic strain | Nmp x 1 | Nmp x 1 |
|  mp_pos  |    particle cordinates    | Nmp x 2 | Nmp x 3 |
|   v_s    |  solid particle velocity  | Nmp x 2 | Nmp x 3 |
|   vol    |   particle volume (tol)   | Nmp x 1 | Nmp x 1 |
|   mass   |    particle mass (tol)    | Nmp x 1 | Nmp x 1 |
|   time   |  current simulation time  |    1    |    1    |


vars for two-phase ↓:

| Var name |         Meaning         |   2D    |   3D    |
| :------: | :---------------------: | :-----: | :-----: |
|  eps_w   |      water strain       | Nmp x 4 | Nmp x 6 |
|   v_w    | water particle velocity | Nmp x 2 | Nmp x 3 |
|    pp    |      pore pressure      | Nmp x 1 | Nmp x 1 |
| porosity | porosity of the medium  | Nmp x 1 | Nmp x 1 |

## 2. Data saved once

|  Var name  |            Meaning             |   2D    |   3D    |
| :--------: | :----------------------------: | :-----: | :-----: |
|  FILE_NUM  |          iteration id          |    1    |    1    |
| grid_init  |       grid node position       | Nn x 2  | Nn x 3  |
|  mp_init   |         pore pressure          | Nmp x 1 | Nmp x 1 |
| vbc_xs_idx | x solid velocity bc node index | vector  | vector  |
| vbc_xs_val | x solid velocity bc node value | vector  | vector  |
| vbc_ys_idx | y solid velocity bc node index | vector  | vector  |
| vbc_ys_val | y solid velocity bc node value | vector  | vector  |
| vbc_zs_idx | z solid velocity bc node index |    -    | vector  |
| vbc_zs_val | z solid velocity bc node value |    -    | vector  |
| vbc_xw_idx | x water velocity bc node index | vector  | vector  |
| vbc_xw_val | x water velocity bc node value | vector  | vector  |
| vbc_yw_idx | y water velocity bc node index | vector  | vector  |
| vbc_yw_val | y water velocity bc node value | vector  | vector  |
| vbc_zw_idx | z water velocity bc node index |    -    | vector  |
| vbc_zw_val | z water velocity bc node value |    -    | vector  |

# Vars saved in VTU (.vtu - ParaView)

| Var name |          Meaning          |
| :------: | :-----------------------: |
|  sig_xx  |         stress xx         |
|  sig_yy  |         stress yy         |
|  sig_zz  |         stress zz         |
|  sig_xy  |         stress xy         |
|  sig_yz  |         stress yz         |
|  sig_zx  |         stress zx         |
|  sig_m   |        mean stress        |
| eps_s_xx |      xx solid strain      |
| eps_s_yy |      yy solid strain      |
| eps_s_zz |      zz solid strain      |
| eps_s_xy |      xy solid strain      |
| eps_s_yz |      yz solid strain      |
| eps_s_zx |      zx solid strain      |
| eps_w_xx |      xx water strain      |
| eps_w_yy |      yy water strain      |
| eps_w_zz |      zz water strain      |
| eps_w_xy |      xy water strain      |
| eps_w_yz |      yz water strain      |
| eps_w_zx |      zx water strain      |
|  v_s_x   |     x solid velocity      |
|  v_s_y   |     y solid velocity      |
|  v_s_z   |     z solid velocity      |
|  v_w_x   |     x water velocity      |
|  v_w_y   |     y water velocity      |
|  v_w_z   |     z water velocity      |
|  disp_x  |      x displacement       |
|  disp_y  |      y displacement       |
|  disp_z  |      z displacement       |
|   disp   |    total displacement     |
|    pp    |       pore pressure       |
| porosity |         porosity          |
|   mass   |       particle mass       |
|   vol    |      particle volume      |
|   epII   | equivlent plastic strain  |
|   epK    | volumetric plastic strain |