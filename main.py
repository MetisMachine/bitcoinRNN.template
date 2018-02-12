# # Bitcoin Price Prediction                                
#   Here we provide a template to precit bitcoin price and deploy   #
#   at scale on a user-defined schedule, taking advantage of        #
#   Metis Machine curated data, and external 3rd party data.        #

## Import some needed dependencies
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import quandl
import torch
import torch.nn as nn
from torch.autograd import Variable


# ## Data Prep
# Specify what data to use from quandl
# Here we grab bitcoin prices & market data
COIN = "BCHARTS/BITSTAMPUSD" 
API = os.environ['QUANDL_KEY']
DATASETS = [COIN, "BCHAIN/CPTRA", "BCHAIN/NTRAT"]

## Set the quandl api key so we can access historical coin data
quandl.ApiConfig.api_key = API
coin_df = quandl.get(DATASETS, returns="pandas")

# We are interested in predicting the close price (*6pm of the next day)
coin_df = coin_df[['BCHARTS/BITSTAMPUSD - Close', 'BCHAIN/NTRAT - Value', 'BCHAIN/CPTRA - Value']].dropna()
coin_df.columns = ['close_price', 'num_trans', 'cost_per_trans']
coin_df_end = coin_df.index.max().date()

# Use the Skafos Data Engine to pull in curated dataset
from skafossdk import *
print('Initializing the SDK connection', flush=True)
skafos = Skafos()

# Query data engine for google keyword trends
res = skafos.engine.create_view(
    "gtrends", {"keyspace": "google_trends", "table": "crypto"}, 
    DataSourceType.Cassandra).result()
print("Created a view of historical google trends data", flush=True)

print("Pulling in historical google trends data...")
gtrends_json = skafos.engine.query("SELECT * from gtrends WHERE keyword IN ('bitcoin', 'blockchain', 'crypto currency', 'litecoin')").result()

# Validate a single record
print("Validating a single record:", flush=True)
print(gtrends_json['data'][0], flush=True)

# Convert to pandas df
gtrends = pd.DataFrame.from_records(gtrends_json['data'])\
    .pivot(index='date', values='interest', columns='keyword')

# Deal with potential nans
for col in gtrends.columns:
    if 'NaN' in gtrends[col].values:
        gtrends[col].replace({'NaN': None}, inplace=True)
    else:
        continue

# If there are nans, fill using pad method
gtrends.fillna(method='pad', inplace=True)

# Set proper date format
gtrends.index = pd.to_datetime(gtrends.index)

# Catch the last date of gtrends data available
gtrends_end = gtrends.index.max().date()

# Figure out how much we might need to shift based on data availability
if (coin_df_end - gtrends_end).days == 0:
    #Same day
    print("Data lined up perfectly, no shifting needed.")
    shifter = 0
elif (coin_df_end - gtrends_end).days == 1:
    # One day behind
    print("Gtrends Data is one day behind. Shifting once.")
    shifter = 1
else:
    # More days behind
    shifter = (coin_df_end - gtrends_end).days
    print("Gtrends Data is %s days behind. Shifting multiple." % shifter)
    
# Join google trends with quandl coin data
df = coin_df.join(gtrends, how='left')


# ## Prep Inputs for Modeling
# We want to use a recurrent time-series model, so our data
# need to be in ascending order by date.
day_zero = df.index.min()
day_index_map = dict(zip((df.index - day_zero).days.values, df.index.values))

df.set_index((df.index - day_zero).days, inplace=True)
df.sort_index(inplace=True)

# Get rid of 0's in price and Calculate percent change in price
df = df[df.close_price != df.close_price.min()]
df['close_price_change'] = df.close_price.pct_change()


# Shift google trends to fill gap in data availability
# NOTE: This means the model is using the search volume some x days prior to
#       prediction. This should account for human lag in research/interest to action.
#       It also may not be as good as one day out or same day ofcourse.
df[['bitcoin', 'blockchain', 'litecoin', 'crypto currency']] = df[['bitcoin', 'blockchain', 'litecoin', 'crypto currency']].shift(shifter)
df.dropna(inplace=True)

# Normalize inputs for deep learning
# Most neural networks expect inputs from -1 to 1
# So we fit two standard deviations in between  -1 and 1
df_scaled = df.apply(lambda c: 0.5 * (c - c.mean()) / c.std())

# Shift so that we're trying to predict tomorrow's price
bitcoin_y = df_scaled['close_price_change'].copy().shift(-1)

bitcoin_x = df_scaled.drop(['close_price'], axis=1)

# Predict on the last day
last_day = max(bitcoin_x.index)


# ## Recurrent Neural Network Model
# [PyTorch](http://pytorch.org) is a wonderful framnework for deep learning 
# since it handles backpropgation automatically.

x_train = torch.autograd.Variable(
    torch.from_numpy(bitcoin_x.loc[:last_day - 1].as_matrix()).float(), requires_grad=False)
x_pred = torch.autograd.Variable(
    torch.from_numpy(bitcoin_x.loc[last_day:].as_matrix()).float(), requires_grad=False)
batch_size = x_train.size()[0]
input_size = len(bitcoin_x.columns)


y_train = torch.autograd.Variable(
    torch.from_numpy(bitcoin_y.loc[:last_day - 1].as_matrix()).float(), requires_grad=False)
y_pred = torch.autograd.Variable(
    torch.from_numpy(bitcoin_y.loc[last_day:].as_matrix()).float(), requires_grad=False)


class CryptoNet(torch.nn.Module):

    def __init__(self, hidden_layers, hidden_size, drop_out_rate):
        super(CryptoNet, self).__init__()
        # set hidden size, layers and dropout rate
        self.drop_out_rate = drop_out_rate
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        # using a GRU (Gated Recurrent Unit), also try an LSTM
        self.rnn1 = nn.GRU(input_size=input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.hidden_layers)
        self.dropout = nn.Dropout(p=self.drop_out_rate)
        self.dense1 = nn.Linear(self.hidden_size, 4)
        self.dense2 = nn.Linear(4, 1)

    def forward(self, x, hidden):
        x_batch = x.view(len(x), 1, -1)
        x_r, hidden = self.rnn1(x_batch, hidden)
        x_d = self.dropout(x_r)
        x_l = self.dense1(x_d)
        x_l2 = self.dense2(x_l)
        return x_l2, hidden

    def init_hidden(self):
        return Variable(torch.randn(self.hidden_layers, 1, self.hidden_size))


# ## Train the RNN

# Setup model for training and prediction
torch.manual_seed(0)
model = CryptoNet(hidden_layers=1, hidden_size=8, drop_out_rate=0.25)
print(model)

# Define loss function and optimizer, tune lr
criterion = nn.MSELoss(size_average=True)
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)

# Initialize the hidden layer during training, but keep it for later prediction.
hidden = model.init_hidden()

# Train the model on 500 epochs
# Ideally this number is tuned precisely
NUM_EPOCHS = 500
for i in range(NUM_EPOCHS):
    def closure():
        model.zero_grad()
        hidden = model.init_hidden()
        out, hidden = model(x_train, hidden)
        loss = criterion(out, y_train)
        if i % 10 == 0:
            print('{:%H:%M:%S} epoch {} loss: {}'.format(datetime.now(), i, loss.data.numpy()[0]), flush=True)
        loss.backward()
        return loss
    optimizer.step(closure)

######################################################

# Predict over the holdout test set and retain the hidden state
pred, new_hidden = model(x_pred, hidden)

def unnormalize(x):
  """Undo the normalization step performed prior to training the model."""
  return (2. * x * df['close_price_change'].std())+df['close_price_change'].mean()

# Unnormalize data and get close price
predicted_value = unnormalize(pred.view(1).data.numpy()[0])
previous_close_price = df.loc[last_day:].close_price.values[0]

# Get the prediction and date value
predicted_price = (predicted_value + 1)*previous_close_price
prediction_date = pd.to_datetime(day_index_map.get(last_day), "%Y-%m-%d").date() \
    + timedelta(days=1)

print("The RNN predicts the closing price for: \n%s to be %s $" % (prediction_date, predicted_price), flush=True)

data_out = [{'price_prediction': predicted_price,
         'date': prediction_date,
         'date_updated': datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"),
         'coin': 'bitcoin'}]


# ## Persist Predictions
# define the schema for this dataset
schema = {
    "table_name": "crypto_predictions",
    "options": {
        "primary_key": ["coin", "date", "date_updated"],
        "order_by": ["date asc"]
    },
    "columns": {
        "coin": "text",
        "date": "date",
        "date_updated": "timestamp",
        "price_prediction": "float"
    }
}

# Save out using the data engine
print("Saving to the data engine.", flush=True)
skafos.engine.save(schema, data_out).result()
print("Done.", flush=True)
