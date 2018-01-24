# bitcoinRNN.template
This template shows basic usage of Metis Machine curated data, 3rd party data, and PyTorch framework to implement a recurrent deep learning model to predict the "closing price" of bitcoin.

## Dependencies
User must aqcuire an API key from [quandl](https://www.quandl.com/) if they wish to use. Otherwise, user can provide their own coin data.

Set the API key using the skafos CLI:
`skafos env QUANDL_KEY --set <API KEY>`

## Model Tuning
The project represents the bare bones of what it takes to build a sophisticated deep learning model. More hyperparameter tuning, feature engineering, and optimizer testing will improve the model's performance.
