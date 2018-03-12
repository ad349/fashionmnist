# fashionmnist
Solution of Kaggle's Fashion MNIST Image classification

| Layer | Description | Shape |
| ----- | ----------- | ----- |
| ConvA | 32, 5x5, 1x1 | (Batch_size, 28, 28, 1) |
| BatchNormA | | (Batch_size, 28, 28, 32) |
| MaxPoolA | 2x2, 2x2 | (Batch_size, 28, 28, 32) |
| ConvB | 64, 5x5, 1x1 | (Batch_size, 14, 14, 32) |
| BatchNormB | | (Batch_size, 14, 14, 64) |
| MaxPoolB | 2x2, 2x2 | (Batch_size, 14, 14, 64) | 
| Fullyconnc | 1024 | (Batch_size, 3136) |
| Softmax | 10 | (Batch_size, 10) |

python train_softmax_clean.py \
--train_csv ./fashionmnist/training2.csv \
--batch_size 100 \
--buffer_size 15000 \
--lr 0.0001 \
--log_dir ./log \
--model_dir ./model \
--nrof_epochs 5

# To Do
- [x] Add Summaries and Plots
- [x] Save checkpoint
- [x] Add validation loss and accuracy
- [ ] Export Model pb
- [ ] Early Stopping
- [ ] Add image summary of activation maps
