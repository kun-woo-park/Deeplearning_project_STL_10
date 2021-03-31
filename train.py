import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchsummary import summary
from function_class_for_train import Model, train_data_load, train_model

# setting number of used gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    ########## Hyper Parmeters ################
    train_batch_size = 128                                              # batch size for train
    fine_tune_batch_size = 64                                           # batch size for fine tune
    num_of_data_augmentation = 100000                                     # number of data augmentation (>= 5000)
    train_lr = 0.001                                                    # learning rate for train
    fine_tune_lr = 0.0005                                                # learning rate for fine tune
    train_total_epoch = 300                                               # total epoch for train
    fine_tune_total_epoch = 140                                           # total epoch for fine tune
    lr_gamma = 0.6                                                      # learning rate reduction rate
    patience = 10                                                       # early stop patience
    start_early_stop_check = 0                                          # early stop start epoch
    saving_start_epoch = 10                                             # starting epoch for saving model
    model_char = "3.0"                                                  # model version
    saving_path = "./"                                                  # saving path
    train_path = f"./data/split_train/train"
    test_path = f"./data/split_train/val"
    T_0 = 10                                                            # starting number of cosine annealing period
    T_mul = 2                                                           # multiplier for cosine annealing period
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],        # normalize factor for STL 10
                                     std=[0.267, 0.256, 0.276])
    #############################################
    val_acc_list = []                                                   # validation accuracy list for print

    train_resolution = 70                                               # train image resolution
    # load train data
    train_loader, val_loader = train_data_load(train_path, test_path, train_batch_size, train_resolution, normalize, num_of_data_augmentation)
    # define model
    model = Model()
    if torch.cuda.is_available():
        model.cuda()
    # print model summary
    summary(model, (3, 96, 96))
    # train model
    train_model(model, train_total_epoch, val_acc_list, train_batch_size, train_lr, lr_gamma, patience, start_early_stop_check,
                saving_start_epoch, model_char, saving_path, T_0, T_mul, train_loader, val_loader)
    # flush gpu memories
    torch.cuda.empty_cache()

    test_resolution = 120                                               # test image resolution
    # load fine tune data
    fine_tune_train_loader, fine_tune_val_loader = train_data_load(train_path, test_path, fine_tune_batch_size, test_resolution, normalize)
    # fine tune model
    train_model(model, fine_tune_total_epoch, val_acc_list, fine_tune_batch_size, fine_tune_lr, lr_gamma, patience, start_early_stop_check,
                saving_start_epoch, model_char, saving_path, T_0, T_mul, fine_tune_train_loader, fine_tune_val_loader)

    # plotting validation accuracy result
    plt.figure(figsize=(15, 15))
    plt.ylabel("val_accuracy")
    plt.xlabel("epoch")
    plt.plot(val_acc_list)
    plt.grid()


