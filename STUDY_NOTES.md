# Study notes: questions you should be able to answer about this code

These are the questions an interviewer is likely to ask after reading this
repository, with the answers grounded in the actual code.

## 1. Why global average pooling instead of flattening the feature maps?

Flattening the final 128x7x7 tensor into a dense layer would add ~800k
parameters on a 4,708-image dataset, which is an overfitting invitation.
Global average pooling reduces each channel to one number, so the classifier
head has 128 inputs and the whole model stays at ~93k parameters. It also
makes the network mostly independent of input size: the same architecture
accepts 28x28 or 224x224 images without changing the classifier.

## 2. How did you handle the class imbalance, and what are the alternatives?

The training split is roughly 3:1 pneumonia to normal. I used
`BCEWithLogitsLoss(pos_weight = neg/pos = 0.35)`, which scales the loss of
the majority (pneumonia) class down so the gradient is not dominated by it.
Alternatives: oversampling the minority class (duplicates rare examples,
more overfitting risk), undersampling (throws away data), or focal loss
(down-weights easy examples). Weighted loss is the least invasive and keeps
the data distribution honest.

## 3. Why report ROC-AUC next to accuracy?

Accuracy depends on an arbitrary 0.5 threshold and is misleading on
imbalanced data (predicting "pneumonia" for everything scores ~75% on this
test set). ROC-AUC measures ranking quality across all thresholds, so it
survives both the imbalance and the threshold choice. In a medical setting I
would also pick the operating threshold on the validation set for a target
recall, because a missed pneumonia case costs far more than a false alarm,
and then report precision/recall at that threshold.

## 4. Walk me through the architecture and the role of batch normalization.

Input 1x28x28 -> conv(1->32, 3x3) + BN + ReLU -> conv(32->64, 3x3) + BN +
ReLU -> maxpool(2) -> conv(64->128, 3x3) + BN + ReLU -> maxpool(2) -> global
average pool -> dropout(0.25) -> linear(128->1). Batch normalization
stabilizes the activation distributions between layers, which lets the small
network train in 6 epochs without careful learning-rate schedules; because BN
uses batch statistics during training and running statistics at eval, it
also adds a mild regularization effect. `bias=False` on the convs because BN
already subtracts the mean, making the conv bias redundant.

## 5. How do you know the model is not overfitting, and what would you do next?

Three signals: the validation loss tracks the training loss instead of
diverging, test metrics are consistent with validation accuracy, and the
model is small relative to the data (93k parameters, ~4.7k training images
with weight sharing in convolutions). The split discipline matters too:
train/val/test come from the dataset as published, and the test set is
touched exactly once. Next steps: transfer learning from a pretrained ResNet
on the full-resolution source images, light augmentation to stabilize the
validation curve (it fluctuates between epochs at this tiny val-set size),
and threshold tuning plus probability calibration before any clinical-sounding
claim.
