# happiness
Sentiment analysis on ~70,000 TripAdvisor hotel reviews using fastai, LSTM, stacking, xgboost

This code makes use of the fastai library found at https://github.com/fastai/fastai to label TripAdvisor hotel reviews as having positive or negative sentiment

Basic steps taken were to create a LSTM language model and feed that plus a few categorical features to a RNN.  After tuning, do 10 fold cross validation, get the out-of-fold probability predictions, average them, and convert to the target labels.  Also tried feeding these 10 outputs along with output from lightgbm and a couple other models to XGBoost, which resulted in a slight improvement from .91051 to .91382 accuracy on a public LB

## Things I'd do differently next time

1. Start with a smaller subset of data.  I forgot to do this and spent too long on parameter tuning as a result.
1. Stop training when overfitting.  I kept trying to train more, thinking I'd reach a new minimum error.  Sometimes it did, but only after many epochs, and I think the gains were basically in the same valley
1. Use algorithms that implement early stopping, or implemenet it myself
