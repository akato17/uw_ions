import numpy as np
import fieldfunc as ff
import userconfig as uc
import gc

#Class version of Field initialization and import
class Field:
    
    def __init__(self, mode = 'create', VRF = np.array([0,0,0,0,0,0,0,0,1,1]), VDC = np.array([0,0,0,0,0,0,0,0,1,1]), paramsRF_file = 'paramsRF.npy', paramsDC_file = 'paramsDC.npy'):
        
        # Initialize by creating from electrode configuration
        if(mode == 'create'):
            
            print('Initializing the field... It takes for a while.')
            
            V0RF, V0DC, cord = ff.trap_potentials(VRF, VDC) # Get V0 of the field

            # Fit with hyperboloid and scale with length in meter
            VRF_fit = np.max(VRF)
            ## Estimate the constant term
            if(abs(np.min(VRF)) > abs(np.max(VRF))):
                VRF_fit = np.min(VRF)
            VDC_fit = np.max(VDC)
            if(abs(np.min(VDC)) > abs(np.max(VDC))):
                VDC_fit = np.min(VDC)
            
            self.paramsRF, self.paramsDC = ff.curve_fit(V0RF, V0DC, cord, VRF_fit, VDC_fit)
            self.paramsRF *= np.append((np.zeros(6)+1) * ((uc.field_config['half_grid_points']*2+1)/(uc.field_config['half_grid_length']*2))**2, [1])
            self.paramsDC *= np.append((np.zeros(6)+1) * ((uc.field_config['half_grid_points']*2+1)/(uc.field_config['half_grid_length']*2))**2, [1])

            # Collect useless temp variables
            del V0RF
            del V0DC
            del cord
            gc.collect()

            print('Initialized!')
        
        # Initialize from existed files
        elif(mode == 'import'):
            self.paramsRF = np.load(paramsRF_file)
            self.paramsDC = np.load(paramsDC_file)
        
        # Report invalid mode
        else:
            raise ValueError('Invalid Mode')
    
    def save_params(self, paramsRF_file = 'paramsRF.npy', paramsDC_file = 'paramsDC.npy'):
        np.save(paramsRF_file, self.paramsRF)
        np.save(paramsDC_file, self.paramsDC)