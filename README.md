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

Set up the environment with the following commands using [uv](      https://github.com/astral-sh/uv): 
```bash
uv venv --python=python3.10 .venv
source .venv/bin/activate 
uv pip install -r requirements.txt
```


## Usage

We provide a simple example of how to use PCS UQ to generate prediction intervals/sets. 
```python
from src.pcs.regression.pcs_oob import PCS_OOB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

models = {"RF": RandomForestRegressor(), "OLS": LinearRegression()}
pcs = PCS_OOB(num_bootstraps = 100, models = models) # initialize the PCS object and provide list of models to fit as well as number of bootstraps

# TODO: add the data

pcs.fit(X, y) # fit the model
pcs.calibrate(X,y) # calibrate the model
pcs.predict(X) # generate prediction intervals/sets
```

