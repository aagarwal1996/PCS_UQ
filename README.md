<h1 align="center"> Uncertainty Quantification via the Predictability, Computability, Stability (PCS) Framework </h1>

<p align="center">  PCS UQ is a Python library for generating prediction intervals/sets via the PCS framework. Experiments in our paper show that PCS UQ reduces average prediction intervals significantly compared to leading conformal inference methods. 

</p>

## Set-up 

### Installation 

```bash
pip install pcs_uq
# Alternatively,  
# clone then pip install -e .
```


### Environment Setup 

Set up the environment with the following commands: 
```bash
conda create -n pcs_uq python=3.10 
pip install -r requirements.txt 
pip install -e . 
```


## Usage

We provide a simple example of how to use PCS UQ to generate prediction intervals/sets. 
```python
from pcs_uq import PCS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

pcs = PCS(num_bootstraps = 100, estimators = [RandomForestRegressor(), LinearRegression()]) # initialize the PCS object and provide list of models to fit as well as number of bootstraps
pcs.fit(X, y) # fit the model
pcs.calibrate(X,y) # calibrate the model
pcs.predict(X) # generate prediction intervals/sets
```

