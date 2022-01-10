import numpy as np
import fieldfunc as ff
import userconfig as uc
import gc

class Field:
    
    def __init__(self, mode = 'create', VRF = np.array([0,0,0,0,0,0,0,0,1,1]), VDC = np.array([1,0,0,0,1,0,0,0,0,0]), paramsRF_file = 'paramsRF1.npy', paramsDC_file = 'paramsDC1.npy'):
        
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
    
    def save_params(self, paramsRF_file = 'paramsRF1.npy', paramsDC_file = 'paramsDC1.npy'):
        np.save(paramsRF_file, self.paramsRF)
        np.save(paramsDC_file, self.paramsDC)
def trap_frequency(V_RF,V_END,V_SIDE):
        """
        Takes aconfiguration of voltagesand calculates the trap frequency
        V_RF is the amplitude of RF appliedto the endcaps
        V_END is the amplitude of DC applied to the endcaps
        V_SIDE is the amplidue of DC applied to electrodes 1 and 5
        the trap frequency in x and y is calculated in a rotated frame 
        currently only supported for electrodes 1 and 5, but more functionality could be added
        you must havesimulated the endcapsat 1V, and the two opposing electrodes 1 and 5 also at 1V
        """
        endcap_params_1V=np.load("1Vendcaps4.npy")
        
        side_params_1V=np.load("1Vsides4.npy")
        
        rf_param=endcap_params_1V*V_RF
        end_param=endcap_params_1V*V_END
        side_param=side_params_1V*V_SIDE
        a1=rf_param[0]
        c1=rf_param[2]
        d1=rf_param[3]
        e1=rf_param[4]
        f1=rf_param[5]
        RF_xterm=uc.Q**2/(2*uc.m**2*uc.omega**2)*(4*a1**2+d1**2+e1**2)
        RF_zterm=uc.Q**2/(2*uc.m**2*uc.omega**2)*(4*c1**2+f1**2+e1**2)
        
        a2=end_param[0]
        c2=end_param[2]
        
        end_xterm=2*uc.Q/uc.m*a2
        end_zterm=2*uc.Q/uc.m*c2
    #####angle of the rotated coordinate system
        theta=22.5*np.pi/180+np.pi/2
        a3=side_param[0]
        b3=side_param[1]
        c3=side_param[2]
        d3=side_param[3]
        
        side_xterm=2*uc.Q/uc.m*(a3*np.cos(theta)**2+b3*np.sin(theta)**2+d3*np.sin(theta)*np.cos(theta))
        side_yterm=2*uc.Q/uc.m*(a3*np.sin(theta)**2+b3*np.cos(theta)**2-d3*np.sin(theta)*np.cos(theta))
        side_zterm=2*uc.Q/uc.m*c3
        
        xterm_sum=RF_xterm+end_xterm+side_xterm
        yterm_sum=RF_xterm+end_xterm+side_yterm
        zterm_sum=RF_zterm+end_zterm+side_zterm
        print(xterm_sum)
        wx=1/(2*np.pi)*np.sqrt(xterm_sum)
        wy=1/(2*np.pi)*np.sqrt(yterm_sum)
        wz=1/(2*np.pi)*np.sqrt(zterm_sum)
        
        return wx,wy,wz
        
        