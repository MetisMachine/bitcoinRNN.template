# bitcoinRNN.template
This template shows basic usage of Metis Machine curated data, 3rd party data, and PyTorch framework to implement a recurrent deep learning model to predict the "closing price" of bitcoin.

Discliamer: Obviously, this model is intended to get you started working with neural networks on the Skafos platform. DO NOT use this example model to "predict" the price of Bitcoin, or any other cryptocurrency asset. Using only past observations and Google trend data leaves lots of room for improvement! See full legal disclaimer [here](https://docs.metismachine.io/docs/predict-the-price-of-cryptocurrency-in-10-minutes).

## Dependencies
User must aqcuire an API key from [quandl](https://www.quandl.com/) if they wish to use. Otherwise, user can provide their own coin data.

Set the API key using the skafos CLI:
`skafos env QUANDL_KEY --set <API KEY>`

## Model Tuning
The project represents the bare bones of what it takes to build a sophisticated deep learning model. More hyperparameter tuning, feature engineering, and optimizer testing will improve the model's performance.
