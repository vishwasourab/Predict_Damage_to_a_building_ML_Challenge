# Predict Damage to a building - ML Challenge on Hackerearth

Determining the degree of damage that is done to buildings post an earthquake can help identify safe and unsafe buildings, thus avoiding death and injuries resulting from aftershocks.  Leveraging the power of machine learning is one viable option that can potentially prevent massive loss of lives while simultaneously making rescue efforts easy and efficient.

In this challenge we are provided with the before and after details of nearly one million buildings after an earthquake. The damage to a building is categorized in __five grades__. Each grade depicts the extent of damage done to a building post an earthquake.

## Goal of the Project:
our task is to __build a model that can predict the extent of damage that has been done to a building after an earthquake.__

## Author:
- [Brungi Vishwa Sourab](https://brungivishwasourab.com)

## Achievements about this project:
- __Rank 92__ out of 7540 on public [leaderboard](https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-6-1/leaderboard). 

## Data Variables Description:

| Variable  | Description |
| ------------- | ------------- |
| area_assesed  |Indicates the nature of the damage assessment in terms of the areas of the building that were assessed  |
| building_id  | A unique ID that identifies every individual building  |
| damage_grade  |Damage grade assigned to the building after assessment (Target Variable)  |
| district_id  | District where the building is located |
| has_geotechnical_risk  |Indicates if building has geotechnical risks  |
| has_geotechnical_risk_fault_crack  | Indicates if building has geotechnical risks related to fault cracking  |
| has_geotechnical_risk_flood  |Indicates if building has geotechnical risks related to flood |
| has_geotechnical_risk_land_settlement  | Indicates if building has geotechnical risks related to land settlement |
| has_geotechnical_risk_landslide |Indicates if building has geotechnical risks related to landslide  |
| has_geotechnical_risk_liquefaction |Indicates if building has geotechnical risks related to liquefaction |
| has_geotechnical_risk_other |Indicates if building has any other  geotechnical risks  |
| has_geotechnical_risk_rock_fall  |Indicates if building has geotechnical risks related to rock fall |
| has_repair_started  |Indicates if the repair work had started |
| vdcmun_id  | Municipality where the building is located |

#### Libraries Used:
```
numpy
pandas
matplotlib
sklearn
keras
```

## Preprocessing Steps done:

#### 1. Merging Datasets:
In the competition, we were given with four datasets: __train.csv, test.csv, Building_Ownership_Use.csv, Building_Structure.csv__. Our first task was to merge the datasets. First of all, I had merged Building_Ownership_Use and Building_Structure datasets on __building_id__ column (lets call this as buildings dataset). Then, merged train dataset with buildings dataset on __building_id__ using __left join__. After Merging train with buildings dataset, I did the same with the test dataset.

Merging the datasets is done using ``` pd.merge``` method.

#### 2. Dealing with Null Values:
Machine Learning models can not accept null values while training the model. So, we have to either remove the null values or fill them. We can use several methods to fill null values. But in numerical datasets, the ```mean & mode``` are most favourite methods.

Null values are filled using ```fillna()``` method in python

#### 3. One-Hot Encode the data:
Machine Learning models performs fast and accurate if the data is __one-hot encoded__.(No official statement but through personal experience). So, I have encoded both train and test datasets. Before encoding, there are 56 columns in train and test (I have already separated target from train) and after encoding, the number of columns in test and train are 97.

One Hot Encoding is done using ``` pd.get_dummies()``` function. After encoding, we need to align the columns in train and test datasets. This is done using ```align()``` method in python.

#### 4. Scaling the data:
We have converted the data to numerical. But, the numerical ranges of each columns are different. This tends our model to __overfit__ easily. We scaled all the columns in our data to the range ```0-1 (Both Inclusive)```.

Scaling is done using ```MinMaxScaler()``` method in ```sklearn.preprocessing``` library.

## Models:

### Neural Network:
 Built a Sqeuential model with 1 input, 4 hidden & output layers in Keras. ```relu``` is used as activation function in input and hidden layers and a ```softmax``` with 5 classes for the output layer. The code used to build the architecture mentioned is 
 ```
 keras_model= Sequential()

keras_model.add(Dense(256,input_shape=X_train.shape[1:],activation='relu'))
keras_model.add(Dense(128,activation='relu'))
keras_model.add(Dense(64,activation='relu'))
keras_model.add(Dense(32,activation='relu'))
keras_model.add(Dense(16,activation='relu'))
keras_model.add(Dense(5,activation='softmax'))
```
We can see the summary of our model using ```summary()``` method in keras. Our model Summary was:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 256)               25088     
_________________________________________________________________
dense_2 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_3 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_4 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_5 (Dense)              (None, 16)                528       
_________________________________________________________________
dense_6 (Dense)              (None, 5)                 85        
=================================================================
Total params: 68,933
Trainable params: 68,933
Non-trainable params: 0
_________________________________________________________________
```
The above model was trained for ```10 epochs``` and this model gave an accuracy of ```71.88%```. 

### Random Forest Model:
The model that gave me the top score is a Random Forest Model. The hyperparameters used are:
```
n_estimators=400
min_samples_split=3
random_state=120
```
*I trained this model with minimal tuning of hyperparameters because of the computational time and costs.*

After training the model, I have checked the importances of features. The top 10 features with importance percentages are:

| Variable  | Importance Percentage |
| ------------- | ------------- |
| height_ft_post_eq  | 12.04 |
| count_floors_post_eq  | 11.13  |
| condition_post_eq_Not damaged  | 5.49  |
| age_building  | 5.07 |
| plinth_area_sq_ft |4.98 |
|ward_id_x	|4.72|
|ward_id_y	|4.6|
|area_assesed_Both |	4.4|
|condition_post_eq_Damaged Not used	| 3.7|
|area_assesed_Building removed	| 3.56|


__Thank You for your time__ 

with :heart:, __Brungi Vishwa Sourab__
