Nearness of Neighbors Attention (NONA) Predictor - a differentiable non-parametric predictor written in pytorch.
Inspired by KNN and self attention, NONA is a task-robust alternative to a final dense layer for deep models.

NONA computes similarities between final embeddings of different samples and then takes a weighted average of true labels to make predictions.

Rather than requiring the final hidden embeddings to be related to the target linearly (or to be linearly separable for classification), NONA requires only a continuous relationship between embeddings and targets.

Preliminary comparative analyses show statistically significant improvement in models that predict with NONA VS dense layers, especially in the context of finetuning feature extractors for predicting continuous targets.
