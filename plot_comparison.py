"""
Plots the loads flowers
"""
import numpy as np
import matplotlib.pyplot as plt

####################################### INPUTS#####################################
design_loads_original = np.array([
    [3.706034e+05, 1.101813e+05, 5.244604e+04, #Extreme loads
    2.423309e+04, -2.305209e+04, -6.270792e+04,
    3.369653e+04, 6.593],
    [4.779218e+04, 2.218892e+04, 1.158065e+04, #fatigue loads
    1.491166e+03, 1.923159e+03, 1.871442e+04,
    1.701559e+04]
])

design_loads_redesigned = np.array([
    [3.942804e+05, 1.565319e+05, 5.701050e+04, #Extreme loads
    2.912175e+04, -2.573848e+04, -7.283247e+04,
    4.263604e+04, 8.747500e+00],
    [5.645567e+04, 3.963797e+04, 1.023125e+04, #Fatigue loads
    1.508731e+03, 1.028497e+03, 1.784370e+04,
    2.101190e+04]
])


design_loads_redesigned2 = np.array([
    [ 3.646420e+05,  1.101813e+05,  4.978893e+04, #Extreme loads
     2.446630e+04, -2.065615e+04,  -6.937121e+04,
     3.088640e+04,  9.401650e+00],
    [4.379754e+04 , 1.885766e+04 ,  9.585024e+03 , #Fatigue loads
     1.246914e+03,  9.654324e+02, 1.685751e+04 ,
     1.999097e+04 ]
])

AEP = np.array([32919, 35553.8,  34253.4]) #[original 10MW, redesigned 10MW] MWh
AEP_percent = []
for i in range(len(AEP)-1):
    percent = (AEP[i+1] - AEP[0])/AEP[0]*100
    AEP_percent.append(percent)
######################################################################################33

comparison_extrem1 = np.divide(design_loads_redesigned[0], design_loads_original[0])
comparison_fatigue1 = np.divide(design_loads_redesigned[1],design_loads_original[1])

comparison_extrem2 = np.divide(design_loads_redesigned2[0], design_loads_original[0])
comparison_fatigue2 = np.divide(design_loads_redesigned2[1],design_loads_original[1])

ylabels= np.array(["Tower-base FA [kNm]",
    "Tower-base SS [kNm]",
    "Yaw-bearing pitch [kNm]",
    "Yaw-bearing roll [kNm]",
    "Shaft torsion [kNm]",
    "OoP BRM [kNm]",
    "IP BRM [kNm]",
    "Tower clearance [m]"])

rads = np.arange(0, (2 * np.pi), 0.01)
r = np.ones(len(rads))

#Extrem
plt.figure(figsize=(10, 6))
plt.subplot(polar=True)
theta = np.linspace(0, 2 * np.pi, len(comparison_extrem1)+1)
comparison_extrem1 = np.append(comparison_extrem1,comparison_extrem1[0])
comparison_extrem2 = np.append(comparison_extrem2,comparison_extrem2[0])
# Arrange the grid into equal parts in degrees
lines, labels = plt.thetagrids(range(0, 360, int(360/len(ylabels))), (ylabels))
# Plot actual sales graph
plt.plot(rads, r, '--k')
plt.plot(theta, comparison_extrem1 ,'r',marker = 'D')
plt.plot(theta, comparison_extrem2 ,'b',marker = 'D')


# Add legend and title for the plot
plt.title("Extreme loads",fontsize = 15)
plt.tight_layout()
plt.legend(['DTU 10MW', "Redesign V1", "Redesign V2"], loc='best')


# #Fatigue
plt.figure(figsize=(10, 6))
plt.subplot(polar=True)
theta = np.linspace(0, 2 * np.pi, len(comparison_fatigue1)+1)
comparison_fatigue1 = np.append(comparison_fatigue1,comparison_fatigue1[0])
comparison_fatigue2 = np.append(comparison_fatigue2,comparison_fatigue2[0])

theta = np.linspace(0, 2 * np.pi, len(comparison_fatigue1))
# Arrange the grid into equal parts in degrees
lines, labels = plt.thetagrids(range(0, 360, int(360/(len(ylabels)-1.2))), (ylabels[0:-1]))
# Plot actual sales graph
plt.plot(rads, r, '--k')
plt.plot(theta, comparison_fatigue1 ,'r',marker = 'D')
plt.plot(theta, comparison_fatigue2 ,'b',marker = 'D')

# Add legend and title for the plot
plt.title("Fatigue loads",fontsize = 15)
plt.tight_layout()
plt.legend(['DTU 10MW', "Redesign V1", "Redesign V2"], loc='best')


## AEP comparison
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.grid()
ax.set_title("IIB AEP")
turbine = ['DTU 10MW', "Redesign V1", "Redesign V2"]
AEP= [AEP[0],AEP[1], AEP[2]]
ax.bar(turbine,AEP, color = ["grey", "red", "blue"] )
ax.set_ylabel('AEP [MWh]')
plt.show()


