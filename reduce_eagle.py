import sys
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import h5py

from sfms import center, calc_rsfr_io, calc_incl, trans

xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02


DIR = './L100_SF_galaxies/snap_028/'

group_cat = np.load( DIR + 'grp_cat.npy', allow_pickle=True ).item()

sub_vel   = np.column_stack( (group_cat['sub_vel_x'], group_cat['sub_vel_y'], group_cat['sub_vel_z']) )
sub_pos   = np.column_stack( (group_cat['sub_pos_x'], group_cat['sub_pos_y'], group_cat['sub_pos_z']) )
sub_grnr  = np.array(group_cat['Grnr'],dtype=int)
sub_Zgas  = np.array(group_cat['Zgas'])
sub_Mstar = np.array(group_cat['Stellar_Mass'])

galaxy = 9358#sub_grnr[1]
print('Doing Galaxy: %s' %galaxy)

this_sub_vel = sub_vel  [np.where(sub_grnr == galaxy)][0]
this_sub_pos = sub_pos  [np.where(sub_grnr == galaxy)][0]
this_Zgas    = sub_Zgas [np.where(sub_grnr == galaxy)][0]
this_Mstar   = sub_Mstar[np.where(sub_grnr == galaxy)][0]

print( np.log10(this_Mstar) )

gas_data  = np.load( DIR + '%s_prt_0.npy' %galaxy, allow_pickle=True ).item()

gas_pos   = np.array(gas_data['Coordinates'      ])
gas_vel   = np.array(gas_data['Velocity'         ])
gas_mass  = np.array(gas_data['Mass'             ])
gas_sfr   = np.array(gas_data['StarFormationRate'])
gas_rho   = np.array(gas_data['Density'          ])
gas_met   = np.array(gas_data['Metallicity'      ])
scf       = np.array(gas_data['scf'              ])
h         = np.array(gas_data['h'                ])
box_size  = np.array(gas_data['boxsize'          ])

this_sub_pos *= scf # Convert from cMpc to Mpc
this_sub_pos *= 1.00E+3 # Convert from Mpc to kpc

rows_without_nan = ~np.isnan(gas_pos).any(axis=1)
print(sum(rows_without_nan))
rows_without_nan = ~np.isnan(gas_pos).all(axis=1)
print(sum(rows_without_nan))

gas_pos  = gas_pos [rows_without_nan]
gas_vel  = gas_vel [rows_without_nan]
gas_mass = gas_mass[rows_without_nan]
gas_sfr  = gas_sfr [rows_without_nan]
gas_rho  = gas_rho [rows_without_nan]
gas_met  = gas_met [rows_without_nan]


gas_pos   *= 3.2408e-22 # Convert from cm to kpc
gas_pos    = center(gas_pos, this_sub_pos)

print(this_sub_pos)

plt.hist2d( gas_pos[:,1], gas_pos[:,2], cmap=plt.cm.Greys, bins=(100,100) )

plt.savefig('Test_pos.pdf')
plt.clf()

gas_vel   *= 1.00E-05 # Convert from cm/s to km/s
gas_vel   -= this_sub_vel

gas_mass  *= 5.00E-34 # Convert from g to Msun

# gas_rho   *= 5.00E-34 # Convert from g to Msun
# gas_rho   /= (3.086E+21**3.00E+00) # Convert from 1/cm^3 to 1/kpc^3
gas_rho   *= xh / mh

OH         = gas_met * (zo/xh) * (1.00/16.00)
Zgas       = np.log10(OH) + 12

ri, ro = calc_rsfr_io(gas_pos, gas_sfr)
ro2    = 2.000E+00 * ro

sf_idx = gas_rho > 1.00E-1 * ( gas_met / 0.002 ) ** (-0.64) 
print(np.mean(1.00E-1 * ( gas_met / 0.002 ) ** (-0.64) ))
print(sum(sf_idx))
# EAGLE sf density threshold: 1.00E-1 * ( Z / 0.002 ) ** (-0.64) from Schaye (2004; 2015)
incl   = calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)

print(incl)

gas_pos  = trans(gas_pos, incl)

gas_rad = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2 + gas_pos[:,2]**2)


mask = (sf_idx)

gas_rad  = gas_rad  [mask]
gas_mass = gas_mass [mask]
gas_sfr  = gas_sfr  [mask]
gas_rho  = gas_rho  [mask]
Zgas     = Zgas     [mask]



plt.hist2d( gas_rad, Zgas, cmap=plt.cm.Greys, bins=(100,100) )

plt.xlabel( 'Radius' )
plt.ylabel( 'log O/H + 12' )

plt.savefig( 'EAGLE_profile.pdf' )