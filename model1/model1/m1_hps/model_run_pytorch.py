"""PyTorch version of Model 1."""

#####from deephyper.search.util import Timer
#####timer = Timer()
#####timer.start("module loading")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  DataLoader
import numpy as np

#import sambaflow.samba.optim as optim
from early_stopping import EarlyStopping
#import matplotlib.pyplot as plt

#from torch_wrapper import load_cuda_vs_knl, benchmark_forward, use_knl, use_cuda  # noqa
#from utils import get_first_gpu_memory_usage

"""
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def validation_step(...):
    self.log('val_loss', loss)

trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss')])
"""
#####from model1.model1.m1_hps.load_data_pytorch import load_data
#####from model1.model1.m1_hps.Data_Loader import dataset
from load_data_pytorch import load_data
from Data_Loader import dataset
#####timer.end("module loading")


SAMBANOVA = False
DEEPHYPER = True
DEVICE    = None
DTYPE     = None

class Model1(nn.Module):
    """Model 1 object."""

    # dtype=float != torch.float

    def __init__(self, config):
        """Initialize the model object."""
        super(Model1, self).__init__()

        samples = 65
        batchSamples = 26

        numInputs = samples * batchSamples
        input_shape = (numInputs,)
        classCount = self.getClassCount()

        self.batchSamples = batchSamples

        device = config['device']
        dtype  = config['dtype']

        dropout1 = config['dropout1']
        dropout2 = config['dropout2']
        dropout3 = config['dropout3']
        dropout4 = config['dropout4']

        # TODO: How to shuffle.
        # Need batch norm and shuffle.
        # Change BatchNorm to LayerNorm on SambaNova.
        self.ln1   = nn.LayerNorm(input_shape).to(device, dtype=dtype)

        #if batchSamples >= 16:
        in_features = numInputs
        out_features = samples * 16
        self.fc1 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout1 = nn.Dropout2d(dropout1).to(device, dtype=dtype) # This started at 0.20

        #if batchSamples >= 12:
        in_features = out_features
        out_features = samples * 12
        self.fc2 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout2 = nn.Dropout2d(dropout2).to(device, dtype=dtype)

        #if batchSamples >= 10:
        in_features = out_features
        out_features = samples * 10
        self.fc3 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout3 = nn.Dropout2d(dropout2).to(device, dtype=dtype)

        in_features = out_features
        out_features = samples * 7
        self.fc4 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout4 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        #if batchSamples >= 5:
        in_features = out_features
        out_features = samples * 5
        self.fc5 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout5 = nn.Dropout2d(dropout3).to(device, dtype=dtype)

        in_features = out_features
        out_features = samples * 2
        self.fc6 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout6 = nn.Dropout2d(dropout3).to(device, dtype=dtype)

        in_features = out_features
        out_features = 90
        self.fc7 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout7 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 90
        self.fc8 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout8 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 48
        self.fc9 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout9 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 24
        self.fc10 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout10 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 12
        self.fc11 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout11 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 12
        self.fc12 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout12 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 12
        self.fc13 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout13 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 5
        self.fc14 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout14 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 5
        self.fc15 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout15 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 5
        self.fc16 = nn.Linear(in_features, out_features).to(device, dtype=dtype)

        in_features = out_features
        out_features = classCount
        self.fc17 = nn.Linear(in_features, out_features).to(device, dtype=dtype)

    def forward(self, x):
        # sourcery skip: inline-immediately-returned-variable
        """
        Do forward pass.

        Args:
            x: represents one sample of the data.
        """
        device = DEVICE
        dtype  = DTYPE

        # Pass data through ln1
        x = self.ln1(x)

        x = self.fc1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        x = self.dropout1(x)

        x = self.fc2(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        x = self.dropout2(x)

        x = self.fc3(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        x = self.dropout3(x)

        x = self.fc4(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        #x = self.dropout4(x)

        x = self.fc5(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        x = self.dropout5(x)

        x = self.fc6(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        x = self.dropout6(x)

        x = self.fc7(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        x = self.dropout7(x)

        x = self.fc8(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        x = self.dropout8(x)

        x = self.fc9(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        #x = self.dropout9(x)

        x = self.fc10(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        #x = self.dropout10(x)

        x = self.fc11(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        #x = self.dropout11(x)

        x = self.fc12(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        #x = self.dropout12(x)

        x = self.fc13(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        #x = self.dropout13(x)

        x = self.fc14(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        #x = self.dropout14(x)

        x = self.fc15(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x).to(device, dtype=dtype)
        #x = self.dropout15(x)

        x = self.fc16(x)
        x = F.relu(x).to(device, dtype=dtype)

        x = self.fc17(x)


        # Apply log_softmax to x
        #print(f"x.shape: {x.shape}")
        output = F.softmax(x, dim=1)
        return output

    def getClassCount(self):
        """
        Return the number of classes.

        This is a binary classification problem.
        """
        return 2


#def train(  args: argparse.Namespace,
def train(  args,
            model: nn.Module,
            optimizer: optim.AdamW,
            X_train,
            Y_train,
            X_valid,
            Y_valid
            ):
    """
    Train the model.

    Args:
        args: argparse.Namespace,
        model: nn.Module,
        optimizer
        X_train,
        Y_train,
        X_valid,
        Y_valid

    Returns:
        history as a dictionary
            history['acc'] = accuracy
            history['losses'] = losses
    """
    global SAMBANOVA
    global DEEPHYPER
    global DEVICE
    global DTYPE

    #numEpochs = args.epochs
    numEpochs  = args['epochs']
    #batch_size = args.batch_size
    batch_size = args['batch_size']
    batch_size = abs(int(batch_size))

    """
    if SAMBANOVA or DEEPHYPER:
        if args.dry_run:
            args.niter = 1
            numEpochs = args.niter
    """

    trainset = dataset(X_train, Y_train)
    #DataLoader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    validset = dataset(X_valid, Y_valid)
    #DataLoader  # shuffle should not matter here.
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.BCELoss()

    #
    # EarlyStopping start
    #

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    patience = args['patience']
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    #
    # EarlyStopping stop
    #




    #forward loop
    losses = []
    accur = []
    for i in range(numEpochs):

        # This steps through batches.
        for _, (x_train, y_train) in enumerate( trainloader ):

            ###################
            # train the model #
            ###################
            # Clear the gradients of all optimized variables
            optimizer.zero_grad()

            #calculate output
            output = model.forward(x_train.to(DEVICE, dtype=DTYPE))

            #calculate loss
            #print(f"output.shape: {output.shape}")
            #print(f"output[0]: {output[0]}")
            #print(f"y_train.shape: {y_train.shape}")
            #print(f"y_train[0]: {y_train[0]}")


            #
            # TODO:
            # Hmm... this is updating on batches instead of epochs.
            #

            # This is a binary classification problem.  The 'output' columns
            # should be Pn, Py.  Only Py is wanted.
            pyIndex = 1
            loss = loss_fn( (output[:, pyIndex]).reshape(-1,1).to(DEVICE, dtype=DTYPE),
                y_train.reshape(-1,1).to(DEVICE, dtype=DTYPE) )

            #backprop
            loss.backward()         #
            optimizer.step()        #

            # record training loss
            train_losses.append(loss.item())

        #
        # EarlyStopping start
        #

        ######################
        # validate the model #
        ######################
        #model.eval() # prep model for evaluation
        for data, y_valid in validloader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.to(DEVICE, dtype=DTYPE))

            # This is a binary classification problem.  The 'output' columns
            # should be Pn, Py.  Only Py is wanted.
            pyIndex = 1
            loss = loss_fn( (output[:, pyIndex]).reshape(-1,1).to(DEVICE, dtype=DTYPE),
                y_valid.reshape(-1,1).to(DEVICE, dtype=DTYPE) )

            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch = i + 1
        n_epochs = numEpochs
        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model, epoch)

        #
        # EarlyStopping stop
        #

#
#
#
        if epoch % 1 == 0:
            #accuracy calculated across all batches/training data.
            # Do I really want to calculate accuracy across so much data?  No.
            # This has been moved from inside the training loop to here so that
            # it is only calculated when it actually gets used.

            output = model.forward( torch.tensor(X_train, dtype=torch.float32).to(DEVICE, dtype=DTYPE) )

            # This is a binary classification problem.  The 'output' columns
            # should be Pn, Py.  Only Py is wanted.
            pyIndex = 1
            predictedCpu = (output[:,pyIndex]).reshape(-1)
            predicted = predictedCpu.to(DEVICE, dtype=DTYPE)

            for k, pred in enumerate( predicted ):
                if pred < 0.5:
                    predicted[k] = 0
                else:
                    predicted[k] = 1

            predictedCpu = predictedCpu.cpu()

            areEqual = np.equal(predictedCpu.detach().numpy(), Y_train) # Finally

            # Get count of True elements in a numpy array
            acc = np.count_nonzero( areEqual ) / len( areEqual )

            losses.append(loss)
            accur.append(acc)
            print(f"epoch: {epoch}\t loss: {loss}\t accuracy: {acc}")

        #
        # EarlyStopping start
        #

        if early_stopping.early_stop:
            print("Early stopping")

            # This is the true epoch number.
            best = early_stopping.bestEpoch

            best = max(1, best)

            # Truncate lists to the best epoch.
            del accur[best:]
            del losses[best:]

            break

        #
        # EarlyStopping stop
        #

    """
    #plotting the loss
    plt.plot(losses)
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('loss')

    #printing the accuracy
    plt.plot(accur)
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Accuracy')
    plt.ylabel('loss')
    """

    history = {}
    history['acc'] = accur
    history['losses'] = losses
    return history



# Track model history.
HISTORY = None


def run(config):
    """
    Run model.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        accuracy (float): The accuracy of the run.
    """
    global HISTORY
    global SAMBANOVA
    global DEEPHYPER
    global DEVICE
    global DTYPE


    try:
        #####timer.start('loading data')
        (x_train, y_train), (x_valid, y_valid) = load_data(config)
        #####timer.end('loading data')


        #####timer.start('preprocessing')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")

        dtype = torch.float if device == "cuda" else torch.float32

        config['device'] = device
        config['dtype']  = dtype

        DEVICE = device
        DTYPE  = dtype

        model = Model1(config).to(device, dtype=dtype)

        # SambaNova conversions.
        #model.bfloat16().float()
        #samba.from_torch_(model)



        # BCELoss is correct for Model1.
        # Moved to train().
        #criterion = nn.BCELoss()

        # setup optimizer
        #optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
        optimizer = config['optimizer'].lower()

        if optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters())
        elif optimizer == 'adam':
            optimizer = optim.Adam(model.parameters())
        else:
            print("ERROR:")
            print("Choose either AdamW or Adam as an optimizer.")
            return 0.0


        # patient early stopping
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)

        #####timer.end('preprocessing')

        #
        # Training
        #
        #####timer.start('model training')

        history = train(config, model, optimizer, x_train, y_train, x_valid, y_valid)

        #####timer.end('model training')

        HISTORY = history

        return history['acc'][-1]

    except Exception as e:
        import traceback
        print('received exception: ', str(e))
        print(traceback.print_exc())
        return 0.0

if __name__ == '__main__':
    """
activation                                           relu
batch_size                                             64
dropout1                                             0.05
dropout2                                             0.05
dropout3                                             0.05
dropout4                                             0.05
embed_hidden_size                                      21
epochs                                                 15
loss                                  binary_crossentropy
omp_num_threads                                        64
optimizer                                            Adam
patience                                               12
proportion                                            0.9
units                                                  10
id                   5a8aee34-e0cf-11eb-8f12-7fbf608d3b5e
objective                                        0.832512
elapsed_sec                                       4532.67
Name: 1331, dtype: object
    """

    config = {
        'units': 10,
        'activation': 'relu',  # can be gelu
        'optimizer':  'AdamW',  # can be AdamW but it has to be installed.
        'loss':       'binary_crossentropy',
        'batch_size': 32,
        'epochs':     40,
        'dropout1':   0.05,
        'dropout2':   0.05,
        'dropout3':   0.05,
        'dropout4':   0.05,
        'patience':   12,
        'embed_hidden_size': 21,    # May not get used.
        'proportion': .90,          # A value between [0., 1.] indicating how to split data between
                                    # training set and validation set. `prop` corresponds to the
                                    # ratio of data in training set. `1.-prop` corresponds to the
                                    # amount of data in validation set.
        'print_shape': 0,           # Print the data shape.
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"***** device: {device} *****")

    accuracy = run(config)
    print('accuracy: ', accuracy)

    """
    import matplotlib.pyplot as plt
    plt.plot(HISTORY['acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    """
