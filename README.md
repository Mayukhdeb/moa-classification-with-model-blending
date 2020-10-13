# moa-classification-with-model-blending

[**[View notebooks on NBViewer]**](https://nbviewer.jupyter.org/github/Mayukhdeb/moa-classification-with-model-blending/tree/main/notebooks/)

This repo would contain a cleaner version of the solution to the kaggle [MoA prediction challenge](https://www.kaggle.com/c/lish-moa/) built by me and [mainakdeb](https://github.com/Mainakdeb). It's not complete yet, but there's lots of interesting stuff in there already. 

Will add documentation and explanations after the competition ends.

## List of things that have been working very well

* Feature selection with `VarianceThreshold`
* Label smoothing with dual loss: Using the `LabelSmoothedLoss` for backprop and the normal `nn.BCEWithLogitsLoss()` for early stopping
* "Repeat" layers which is not a real thing, but it works well. It's similar to residual layers with `layer_size = (880,880)`
* Single model, multiple fold preds blending with optuna using OOF preds as a metric for loss 


## What can be done 
* Try and incorporate the old 2nd model with residual layers 
* Use optuna instead of using `lr = 4e-3` and `decay_rate = 0.1` and praying that it works
