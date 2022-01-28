# SMP-predictor-Google-stock
Stock Market Prediction using deep learning LSTM method

# Stock Market Predictor (SMP): Predicting a closing price of a stock across a given period of time using deep learning


SMP demonstrates how to use time series data to be able to predict future stock prices. In this project, I predict the prices of Google using an RNN method known as LSTM. 

# Installation to be done in Google colaboratory and [Neptune](https://neptune.ai/)
[Neptune](https://neptune.ai/) is essential for this experiment to import data, monitor and keep track of logs and performance in a more efficient way. 

## Implement all the following in colab 
This is done to ensure that this runs on different environments and devices without installing anything locally that is cumbersome. 

1) Install the updated version of Folium for better visualizations. 

```python
!pip install folium
```
2) Installing the neptune client and importing neptune 

```python
!pip install neptune-client
import neptune.new as neptune 
```
3) Go to [neptune.com](https://ui.neptune.ai/auth/realms/neptune/protocol/openid-connect/registrations?client_id=neptune-frontend&redirect_uri=https%3A%2F%2Fapp.neptune.ai%2F-%2Fonboarding&state=0aaafa39-9505-4146-94c7-79b2d4999788&response_mode=fragment&response_type=code&scope=openid&nonce=1cd4fcb7-fa69-425e-8398-97808feb1012), sign up and create an account

4) Create a project in Neptune, give it a name and a key. Ignore other characteristics of the project for now. Refer to [this](https://docs.neptune.ai/administration/projects#create-project) for more on how to create a project in Neptune. 

5) Get your API token from Neptune. Refer to [this](https://docs.neptune.ai/getting-started/installation#get-and-set-api-token) simple process to get the API. Just get the API token and be ready to use it later in the experiment. Don't care about anything else. 

6) Click on your project properties and get also your project name and key to be used later in this experiment. 


7)  Initialise the project this way: 

- On your Neptune project, click on 'settings', then 'properties' to get the username, project, project name, key, etc...
 - replace 'INPUT USERNAME' with your Neptune username
- replace 'INPUT USERNAME/PROJECT' with your yourusername/project from Neptune 
- replace 'NEPTUNE_API_TOKEN' with your API token from Neptune. 

```python
# Note: You should not worry about GPU metrics error not being reported at this point 

import os

# Connect your script to Neptune new version 
import os
myProject = 'gaelkbertrand/google-stock-prediction'

neptune.init(project=myProject, 
             api_token='INPUT YOUR API TOKEN HERE')

# The code above will also be running neptune to log runs. Don't let the colab runtime run out because it will destroy this process. 

```

## Alas! You are done setting up and ready to run the ML experiment. 

Setting a working directory in Google Drive (to store your data and other logs)

```python
# This helps you setup your working directory to a folder in your Google Drive.

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

#Verify your current directory 
! ls /content/drive/My\ Drive/

import os

# the base Google Drive directory
root_dir = "/content/drive/My Drive/"

# choose where you want your project files to be saved
ML_final_project = "YOUR DESIRED WORKING DIRECTORY"

# check if your project folder exists. if not, it will be created.
def create_and_set_working_directory(ML_final_project):  
  if os.path.isdir(root_dir + ML_final_project) == False:
    os.mkdir(root_dir + ML_final_project)
    print(root_dir + ML_final_project + ' did not exist but was created.')

  # change the OS to use your project folder as the working directory
  os.chdir(root_dir + ML_final_project)

  # create a test file to make sure it shows up in the right place
  !touch 'new_file_in_working_directory.txt'
  print('\nYour working directory was changed to ' + root_dir + ML_final_project + \
        "\n\nAn empty text file was created there. You can also run !pwd to confirm the current working directory." )

create_and_set_working_directory(ML_final_project)

```

## For the following steps, you can just refer to the notebook file and run it after modifying a few necessary things. The things you need to change are specified in the code below. The rest is just about running the notebook as it is. The processes in this note book involve collecting data, training, testing, and collecting performance metrics. The description and steps are ordered as following: 

1. Installing all other libraries and dependencies responsible for machine learning processes. 

2. Importing data from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

- register on Alpha Vantage 
- immediately get a free API key to use in the following code
- know the ticker symbol for a company you need to predict the stock for. For example, here we predict stock prices for google with the ticker symbol 'GOOGL'. Find other companies' ticker symbols [here](https://business.unl.edu/outreach/econ-ed/nebraska-council-on-economic-education/student-programs/stock-market-game/documents/Top%202000%20Valued%20Companies%20with%20Ticker%20Symbols.pdf)

- after running the following step in colab, you can check the data set file stored in your google drive working directory that you specified above. 

```python
#import data from 'Alpha Vantage'

data_source= 'alphavantage'

if data_source == 'alphavantage':
    
    api_key = 'PASTE YOUR ALPHA VANTAGE API KEY HERE'
    # here, write your desired stock ticker symbol
    ticker = 'GOOGL' 
    
    # This is the JSON file with all the stock prices data. Don't change anything here.  
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

    # Save data to this file. Don't change anything here.
    fileName = 'stock_market_data-%s.csv'%ticker

    ### get the low, high, close, and open prices 
    if not os.path.exists(fileName):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # pull the desired stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for key,val in data.items():
                date = dt.datetime.strptime(key, '%Y-%m-%d')
                data_row = [date.date(),float(val['3. low']),float(val['2. high']),
                            float(val['4. close']),float(val['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        df.to_csv(fileName)

    else:
        print('Loading data from local')
        df = pd.read_csv(fileName)
```

3) Data preprocessing 

4) Define functions that helps in calculating the performance: RMSE (Root Mean Square Error), MAPE (Mean absolute percentage error)

5) Splitting the data into a training and test set

6) Creating the LSTM Model
- Running the moving average (MA) step. Read more in the report
- Defining the model, training, and testing 

7) Evaluating the performance. 
- Calculate RMSE and MAPE
- Output prediction graphs 

8) Stop the experiment as follows: 

```python
npt_exp.stop()
```

9) Gpo back to your neptune dashboard and click on the project to view the perfomance metrics and graphs. 

10) Thank you, hope you learned a lot. Key take aways: 

- Running the whole experiment simply on cloud using colab and neptune. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
