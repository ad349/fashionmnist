# fashionmnist
Solution of Fashion MNIST Image classification

python train_softmax_clean.py \
--train_csv ./fashionmnist/training2.csv \
--batch_size 100 \
--buffer_size 15000 \
--lr 0.0001 \
--log_dir ./log \
--nrof_epochs 5

# To Do
* Add Summaries and Plots
* Exponential Learning Rate Decay
* Export Model
* Add validation loss and accuracy
* Prevent Overfitting
