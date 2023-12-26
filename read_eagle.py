import sys
import h5py
import numpy as np
EAGLE_SQL_TOOLS = '/home/alexgarcia/github/eagleSqlTools'
sys.path.insert(1,EAGLE_SQL_TOOLS)
import eagleSqlTools as sql
con = sql.connect( "rbs016", password="yigERne4" )

from sfms import sfmscut
from tqdm import tqdm

import mpi4py as MPI
# MPI.Init()

from os import path, mkdir

DIR  = '/orange/paul.torrey/EAGLE/'
run  = 'RefL0100N1504' + '/'

def get_SF_galaxies(snap, m_star_min=8.0, m_star_max=10.5, m_gas_min=8.5):
    # Create Query of EAGLE database
    myQuery = '''SELECT \
        SF_Metallicity as Zgas,\
        Stars_Metallicity as Zstar,\
        StarFormationRate as SFR,\
        MassType_Star as Stellar_Mass,\
        MassType_Gas as Gas_Mass,\
        HalfMassRad_Gas as R_gas,\
        HalfMassRad_Star as R_star,\
        GroupNumber as Grnr,\
        CentreOfMass_x as sub_pos_x,\
        CentreOfMass_y as sub_pos_y,\
        CentreOfMass_z as sub_pos_z,\
        Velocity_x as sub_vel_x,\
        Velocity_y as sub_vel_y,\
        Velocity_z as sub_vel_z,\
        HalfMassRad_Star as RSHM\
    FROM \
        RefL0100N1504_SubHalo as SH\
    WHERE \
        SnapNum = %s \
        and SH.SubGroupNumber = 0 
        and SH.StarFormationRate > 0.0\
        and SH.MassType_Star > 1E8\
        and SH.SubGroupNumber = 0''' %(snap) 

    print('Starting Query... snapshot %s' %snap)
    sys.stdout.flush()
    myData = sql.execute_query(con, myQuery)
    print('Query Complete')
    sys.stdout.flush()
    
    # Select SF galaxies
    gas_mass  = np.array(myData['Gas_Mass'][:])
    star_mass = np.array(myData['Stellar_Mass'][:])
    SFR       = np.array(myData['SFR'][:])
    
    sfms_idx = sfmscut(star_mass, SFR, m_star_min=8.0, m_star_max=10.5, m_gas_min=8.5)
    
    SFG_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                (star_mass < 1.00E+01**(m_star_max)) &
                (gas_mass  > 1.00E+01**(m_gas_min))  &
                (sfms_idx))
    
    saveData = {}
    
    # You can change these, but need to be changed above accordingly!
    keys = ['Zgas','Zstar','SFR','Stellar_Mass','Gas_Mass',
            'R_gas','R_star','Grnr','sub_pos_x','sub_pos_y','sub_pos_z',
            'sub_vel_x','sub_vel_y','sub_vel_z','RSHM']
    
    # Save only star forming galaxies
    for key in keys:
        saveData[key] = np.array(myData[key])[SFG_mask]
    
    return saveData

def read_eagle_subhalo(file_path, group_index, snap, keys=['GroupNumber','SubGroupNumber'],
                       PartType=0, nfiles=256, sub_group_index=0, file_ext='z000p000'):
    # These parameters are necessary for matching particle data to your galaxy
    req_keys = ['GroupNumber','SubGroupNumber']
    for _k in req_keys:
        if _k not in keys:
            keys.append(_k)
        
    return_dic = {}
    
    galaxy_file_locator = np.load( './L100_SF_galaxies/snap_%s/file_lookup.npy' %str(snap).zfill(3) ,
                                  allow_pickle=True).item()
    
    files_to_look = galaxy_file_locator[group_index] 
    
    for file_counter in files_to_look:
        fname = file_path + 'snap_%s_%s.%s.hdf5' % (str(snap).zfill(3), file_ext, file_counter)
        with h5py.File(fname, 'r') as f:
            a       = f['Header'].attrs.get('Time')
            BoxSize = f['Header'].attrs.get('BoxSize')
            z       = (1.00E+00 / a - 1.00E+00)
            h       = f['Header'].attrs.get('HubbleParam')
            
            # Check to see if this galaxy is in this file
            
            pt = 'PartType' + str(int(PartType))
            
            this_grnrs  = np.array(f[pt]['GroupNumber'])
            this_subnrs = np.array(f[pt]['SubGroupNumber'])
            
            subhalo_mask = ( (this_grnrs  == group_index) &
                             (this_subnrs == sub_group_index) )
    
            subhalo_indices   = np.where(subhalo_mask)[0]
            subhalo_mask_bool = np.zeros_like(subhalo_mask, dtype=bool)
            subhalo_mask_bool[subhalo_indices] = True

            data_dict = {key: mask_array(f[pt][key], subhalo_mask_bool, key) for key in keys}

            for key in keys:                    
                cgs = f[pt][key].attrs.get('CGSConversionFactor')
                axp = f[pt][key].attrs.get('aexp-scale-exponent')
                hxp = f[pt][key].attrs.get('h-scale-exponent')

                if key not in return_dic:
                    return_dic[key] = np.multiply(np.array(data_dict[key]), cgs * a**axp * h ** hxp )
                else:
                    return_dic[key] = np.concatenate((return_dic[key],
                                                      np.multiply(np.array(data_dict[key]), cgs * a**axp * h ** hxp )),
                                                      axis = 0
                                                    )            
        
    # Get header parameters
    with h5py.File(fname, 'r') as f:
        return_dic['scf']     = a      
        return_dic['boxsize'] = BoxSize
        return_dic['z']       = z      
        return_dic['h']       = h      
        
    return return_dic

def mask_array(data, mask, key, chunk_size=5000):
    if (data.ndim > 1):
        result = np.empty_like(data)
        for i in range(0, data.shape[0], chunk_size):
            chunk_data = data[i:i+chunk_size]
            result[i:i+chunk_size] = np.where(mask[i:i+chunk_size, None], chunk_data, np.nan)
    else:
        result = np.where(mask, data, np.nan)        
    return result
    
def get_which_files(file_path, group_indeces, snap, nfiles=256,
                    file_ext='z000p000',save_loc=''):
    all_files = {}

    for file_counter in tqdm(range(nfiles)):
        file_name = file_path + 'snap_{0}_{1}.{2}.hdf5'.format(str(snap).zfill(3), file_ext, file_counter)
        with h5py.File(file_name, 'r') as f:
            subgroup_mask = f['PartType0']['SubGroupNumber'][:] == 0
            unique_gals   = np.unique(f['PartType0']['GroupNumber'][subgroup_mask])

            overlap_gals  = np.intersect1d(unique_gals, group_indeces, assume_unique=True)
            
            for gal in overlap_gals:
                if gal not in all_files:
                    all_files[gal] = [file_counter]
                else:
                    all_files[gal].append(file_counter)

    all_files = {gal: np.array(files) for gal, files in all_files.items()}
    np.save(save_loc, all_files)
    
def save_data(snap, EAGLE, file_ext='z000p000'):
    # Note that I am only interested in central galaxies here... can be modified to include satellite
    
    save_dir = 'L100_SF_galaxies/' + 'snap_%s' %str(snap).zfill(3) + '/' 
    
    # Get SF galaxies at this snapshot
    SF_galaxies = get_SF_galaxies(snap)
    
    # Save the group catalog info
    np.save(save_dir + 'grp_cat' + '.npy', SF_galaxies)
        
        
    subset = SF_galaxies['Grnr'][:10]
        
    # get_which_files(EAGLE, subset, snap, 
    #                 save_loc='./L100_SF_galaxies/snap_%s/file_lookup.npy' %str(snap).zfill(3))
    
    # Loop over all galaxies
    ##########################################
    ## To do: all SF galaxies
    ##########################################
    for galaxy in subset:
        print('Saving Particle Information, central of FoF: %s' %galaxy)
        fname = save_dir + '%s_prt_0' %galaxy + '.npy'
        
        # Save the data
        gas_data  = read_eagle_subhalo(EAGLE, galaxy, snap, PartType=0, file_ext=file_ext,
                                       keys=['Coordinates','Mass','Metallicity',
                                             'Density','StarFormationRate','Velocity'])

        np.save(fname, gas_data)
            
if __name__ == "__main__":
    snap_to_file_name = {
        12:'z003p017',
        15:'z002p012',
        19:'z001p004',
        28:'z000p000'
    }
    
    snap  = 28
    EAGLE = DIR + run + 'snapshot_%s_%s' %(str(snap).zfill(3),snap_to_file_name[snap]) + '/' 
    
    save_data( snap, EAGLE, file_ext=snap_to_file_name[snap] )
