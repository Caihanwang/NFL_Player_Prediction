# Prediction of NFL Player

## Questions:

1. What is the average height and weight for each position?
2. What is the best position for someone who is 5'10" and 155 lbs?
3. For any given height and weight, what is the best position?

## Data

data file `nfl_players.csv`.

---

# Report by Caihan 
We got the basic data of each player in NFL, including their height, weight and position. 

The raw data looks like:

| full_name           | number | position         | height (in) | weight (lb) | date_of_birth | team | sign      |
|---------------------|--------|------------------|-------------|-------------|---------------|------|-----------|
| Alford, Robert      | 23     | cornerback       | 70          | 186         | 11/1/88       | ATL  | scorpio   |
| Babineaux, Jonathan | 95     | defensive tackle | 74          | 300         | 10/12/81      | ATL  | libra     |
| Davis, Dominique    | 4      | quarterback      | 75          | 210         | 7/17/89       | ATL  | cancer    |
| Goodman, Malliciah  | 93     | defensive end    | 76          | 276         | 1/4/90        | ATL  | capricorn |  

We need to create a model using this data and predict the best position for any given height and weight.
<br>  
<br>

## Quesition 1. What is the average height and weight for each position?
The crucial part of my solution is as following:
```python
aver_height=[]
aver_weight=[]
for position in Positions:
    position_data = data[data["position"]==position]
    height = round(np.mean(position_data["height (in)"]),2)
    weight = round(np.mean(position_data["weight (lb)"]),2)
    aver_weight.append(weight)
    aver_height.append(height)
```
The result is as following:
| Positions        | Average Height (in) | Average Weight (lb) |
|------------------|---------------------|---------------------|
| cornerback       | 71.41               | 193.39              |
| defensive end    | 76.03               | 283.18              |
| defensive tackle | 74.85               | 309.77              |
| quarterback      | 75.21               | 223.76              |
| running back     | 70.62               | 215.27              |
<br>    
<br>  

## Question 2. What is the best position for someone who is 5'10" and 155 lbs?
I choose linear discriminant analysis to fit predict model. The crucial part of my solution is as following:
```python
hei_wei = np.array(data[["height (in)","weight (lb)"]])
posi = np.array(data["position"])
clf = LinearDiscriminantAnalysis()
clf.fit(hei_wei, posi)
print(clf.predict([[70,155]]))
```
The result is 'cornerback'.  
<br>  
<br>  

## Question 3. For any given height and weight, what is the best position?  
The crucial part of my solution is as following:
```python
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
```

The result is as following:  

![NFL Player.png](https://i.loli.net/2021/03/01/nEAK7HZyqksSwzB.png)


