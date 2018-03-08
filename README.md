# fashionmnist
Solution of Fashion MNIST Image classification

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
- [ ] Exponential Learning Rate Decay, Stepwise
- [ ] Export Model pb
- [ ] Add validation loss and accuracy
- [ ] Calculate relative improvement and stop training if validation accuracy starts decreasing
- [ ] Add image summary of activation maps
