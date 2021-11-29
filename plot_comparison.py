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

AEP = np.array([32919, 35553.8]) #[original 10MW, redesigned 10MW] MWh

######################################################################################33

comparison_extrem = np.divide(design_loads_redesigned[0], design_loads_original[0])
comparison_fatigue = np.divide(design_loads_redesigned[1],design_loads_original[1])

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
theta = np.linspace(0, 2 * np.pi, len(comparison_extrem)+1)
comparison_extrem = np.append(comparison_extrem,comparison_extrem[0])
# Arrange the grid into equal parts in degrees
lines, labels = plt.thetagrids(range(0, 360, int(360/len(ylabels))), (ylabels)) 
# Plot actual sales graph
plt.plot(theta, comparison_extrem ,'r',marker = 'D')
plt.plot(rads, r, '--b')
 
# Add legend and title for the plot
plt.title("Extrem loads",fontsize = 15)
plt.tight_layout()


# #Fatigue
plt.figure(figsize=(10, 6))
plt.subplot(polar=True)
theta = np.linspace(0, 2 * np.pi, len(comparison_fatigue)+1)
comparison_fatigue = np.append(comparison_fatigue,comparison_fatigue[0])

theta = np.linspace(0, 2 * np.pi, len(comparison_fatigue))
# Arrange the grid into equal parts in degrees
lines, labels = plt.thetagrids(range(0, 360, int(360/(len(ylabels)-1.2))), (ylabels[0:-1]))
# Plot actual sales graph
plt.plot(theta, comparison_fatigue ,'r',marker = 'D')
plt.plot(rads, r, '--b')
 
# Add legend and title for the plot
plt.title("Fatigue loads",fontsize = 15)
plt.tight_layout()


## AEP comparison
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_title("IIB AEP")
turbine = ['Original 10MW [MWh]', "Redesigned 10MW [MWh]"]
AEP= [AEP[0],AEP[1]]
ax.bar(turbine,AEP, color = ["blue", "orange"] )
plt.show()

