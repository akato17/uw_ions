import numpy as np
import fieldfunc as ff
import simfunc as sf
import gc

sample_voltage = np.sin(2 * np.pi * np.linspace(0,13,14) / 14)
print(sample_voltage)

dc_voltage = np.linspace(3,29,14) * 10
print(dc_voltage)

for i in range(0,14):
    print(i)
    RF = np.array([0,0,0,0,0,0,0,0,sample_voltage[i],sample_voltage[i]])
    DC = np.array([dc_voltage[i],0,0,0,dc_voltage[i],0,0,0,0,0])
    V0RF, V0DC, cord = ff.trap_potentials(RF, DC)
    paramsRF, paramsDC = ff.curve_fit(V0RF, V0DC, cord, sample_voltage[i], dc_voltage[i])
    del V0DC
    del V0RF
    gc.collect()
    np.save('paramsRF'+str(i)+'.npy',paramsRF)
    np.save('paramsDC'+str(dc_voltage[i])+'.npy',paramsDC)