"""Analyze nfl player data"""

# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# read data
data = pd.read_csv(
    "./nfl_players.csv"
)



'''Question 1. What is the average height and weight for each position'''

# use loop to calculate average height and weight in each position
Positions = [
    "cornerback", 
    "defensive end", 
    "defensive tackle", 
    "quarterback", 
    "running back"
]

aver_height=[]
aver_weight=[]
for position in Positions:
    position_data = data[data["position"]==position]
    height = round(np.mean(position_data["height (in)"]),2)
    weight = round(np.mean(position_data["weight (lb)"]),2)
    aver_weight.append(weight)
    aver_height.append(height)
# create a data frame to report the result
aver_data_position = pd.DataFrame({
    "Positions" : Positions,
    "Average Height (in)" : aver_height,
    "Average Weight (lb)" : aver_weight
})
print(aver_data_position)



'''Question 2. What is the best position for someone who is 5'10" and 155 lbs?'''

# use linear discriminant analysis to fit model
# extract height, weight, position data to train the model
hei_wei = np.array(data[["height (in)","weight (lb)"]])
posi = np.array(data["position"])
clf = LinearDiscriminantAnalysis()
clf.fit(hei_wei, posi)

print(clf.predict([[70,155]]))



'''Question 3. For any given height and weight, what is the best position? Create a plot'''

# plot the decision boundary by assigning a color in the color map
# create a grid
min1, max1 = hei_wei[:,0].min()-1, hei_wei[:,0].max()+1
min2, max2 = hei_wei[:,1].min()-1, hei_wei[:,1].max()+1
x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)
#predict each point in grid
xx, yy = np.meshgrid(x1grid, x2grid)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# covert string to number
Z[Z=='cornerback']=1
Z[Z=='defensive end']=2
Z[Z=='defensive tackle']=3
Z[Z=='quarterback']=4
Z[Z=='running back']=5
# put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, Z, cmap='twilight')

# plot the scatter
# create a color map and marker map
cmap_bold =['#AFAFAF', '#00BFFF', '#9400D3', '#FF0000', '#F0E68C']
markermap = ["o","^","s","p","d"]
type1_hei = []; type1_wei = []
type2_hei = []; type2_wei = []
type3_hei = []; type3_wei = []
type4_hei = []; type4_wei = []
type5_hei = []; type5_wei = []
# use loop to classify by position
for j in range(len(Positions)):
    for i in range(len(posi)):
        if posi[i] == Positions[j]:
            eval(f"type{j+1}_hei").append(hei_wei[i][0])
            eval(f"type{j+1}_wei").append(hei_wei[i][1])
# plot the scatter of each position
for j in range(len(Positions)):
    plt.scatter(
        eval(f"type{j+1}_hei"),eval(f"type{j+1}_wei"),
        edgecolor = 'black', 
        label = Positions[j], 
        color =cmap_bold[j],
        marker = markermap[j]
    )

plt.legend()
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel("height (in)")
plt.ylabel("weight (lb)")
plt.title("NFL Player's Position and Height&Weight")
plt.savefig("NFL Player")


