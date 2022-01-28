# SMP-predictor-Google-stock
Stock Market Prediction using deep learning LSTM method

# Stock Market Predictor (SMP): Predicting a closing price of a stock across a given period of time using deep learning


SMP demonstrates how to use time series data to be able to predict future stock prices. In this project, I predict the prices of Google using an RNN method known as LSTM. 

## Installation to be done in Google colaboratory and [Neptune](https://neptune.ai/)

1) Install the updated version of Folium for better visualizations. 

```bash
!pip install folium
```
2) Installing neptune client and importing neptune 

```bash
!pip install neptune-client
import neptune.new as neptune 

#initialising the project 
#Note: You should not worry about GPU metrics not being reported at this point 
workspace = 'INPUT USERNAME' # username = 'common' for ANONYMOUS users
project = 'INPUT USERNAME/PROJECT'
project_name = "INPUT PROJECT NAME"

# Connect your script to Neptune new version 
import os
myProject = 'INPUT USERNAME/PROJECT'
project = neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN')
                       project=myProject) 
project.stop()

#running neptune to log runs 
run = neptune.init(project=project_name, api_token=api_token)

```

3) Setting a working directory in Google Drive

```

```
4) Importing needed data

```

```

5) Preprocess data

```

```

5) Calculate RMSE and MAPE

```

```

5) The model

```

```

5) Performance 

```

```


## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
