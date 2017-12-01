# happiness
Sentiment analysis on ~70,000 TripAdvisor hotel reviews using fastai, LSTM, stacking, xgboost

This code makes use of the fastai library found at https://github.com/fastai/fastai to label TripAdvisor hotel reviews as having positive or negative sentiment

Basic steps taken were to create a LSTM language model and feed that plus a few categorical features to a RNN.  After tuning, do 10 fold cross validation, get the out-of-fold probability predictions, average them, and convert to the target labels.  Also tried feeding these 10 outputs along with output from lightgbm and a couple other models to XGBoost, which resulted in a slight improvement from .91051 to .91382 accuracy on a public LB

## Things I'd do differently next time

1. Start with a smaller subset of data.  I forgot to do this and spent too long on parameter tuning as a result.
1. Stop training when overfitting.  I kept trying to train more, thinking I'd reach a new minimum error.  Sometimes it did, but only after many epochs, and I think the gains were basically in the same valley
1. Use algorithms that implement early stopping, or implement it myself

## Description of files
* `Happiness-cv10.ipynb` - does training, saves models
* `Happiness-cv-predict.ipynb` - Basically same code as previous file, but set up to run predictions using a different GPU so it can be done in parallel with training
* `Happiness-stack.ipynb` - A few different attempts at stacking.  This part was after the competition ended.

### More on Stacking

I was trying get this to work under a time constraint (an hour), and therefore tried a few different implementations that I saw people recommended.  Given an hour, I couldn't find the right parameters to start Vowpal Wabbit for training.  Its training kept finishing in 1 second and I felt that couldn't be right.

I also tried Driverless AI, which is commercial and too expensive for individuals (quoted $80k/year, I have a trial now), and it performed as well on the merged predictions as my tuning in XGBoost.  I didn't want to submit this result before the end of the competition since DAI's software is proprietary.

I think DAI would be a good option for organizations who want to do deep learning, can't find data scientists, and want to do some of the work with their own staff rather than handing the whole process to consultants.

Ultimately, XGBoost was easiest to set up thanks to [this walkthrough](https://jessesw.com/XG-Boost/), whose initial parameters and tuning process seemed to work well for this dataset.

## Prep for next time

I want to get familiar with more automated deep learning tools.  StackNet and Xcessive are at the top of my list.  After doing parameter tuning once for one competition, I'm eager to reduce time spent on that

## Final thoughts

Deeping learning is both easy and hard.  Easy because you can learn it, hard because improving on results with parameter tuning can take a lot of time, and it's easy to follow rabbit holes that yield marginal gains.  Shifting gears to try a new approach that could yield more gains takes more effort.
