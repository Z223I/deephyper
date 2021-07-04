"""PyTorch version of Model 1."""

from deephyper.search.util import Timer
timer = Timer()
timer.start("module loading")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  DataLoader
#import sambaflow.samba.optim as optim
#import EarlyStopping
import matplotlib.pyplot as plt

"""
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def validation_step(...):
    self.log('val_loss', loss)

trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss')])
"""
from model1.model1.m1_hps.load_data_pytorch import load_data
from model1.model1.m1_hps.DataLoader import dataset
timer.end("module loading")


SAMBANOVA = False
DEEPHYPER = True

class Model1(nn.Module):
    """Model object."""

    def __init__(self, config):
        """Initialize the model object."""
        super(Model1, self).__init__()

        samples = 65
        batchSamples = 26

        numInputs = samples * batchSamples
        input_shape = (numInputs,)
        classCount = self.getClassCount()

        self.batchSamples = batchSamples

        # TODO: How to shuffle.
        # Need batch norm and shuffle.
        # Change BatchNorm to LayerNorm on SambaNova.
        self.ln1   = nn.LayerNorm(input_shape)

        #if batchSamples >= 16:
        in_features = numInputs
        out_features = samples * 16
        self.fc1 = nn.Linear(in_features, out_features)
        self.dropout1 = nn.Dropout2d(0.20)

        #if batchSamples >= 12:
        in_features = out_features
        out_features = samples * 12
        self.fc2 = nn.Linear(in_features, out_features)
        self.dropout2 = nn.Dropout2d(0.10)

        #if batchSamples >= 10:
        in_features = out_features
        out_features = samples * 10
        self.fc3 = nn.Linear(in_features, out_features)
        self.dropout3 = nn.Dropout2d(0.10)

        in_features = out_features
        out_features = samples * 7
        self.fc4 = nn.Linear(in_features, out_features)
        #self.dropout4 = nn.Dropout2d(0.00)

        #if batchSamples >= 5:
        in_features = out_features
        out_features = samples * 5
        self.fc5 = nn.Linear(in_features, out_features)
        self.dropout5 = nn.Dropout2d(0.05)

        in_features = out_features
        out_features = samples * 2
        self.fc6 = nn.Linear(in_features, out_features)
        self.dropout6 = nn.Dropout2d(0.05)

        in_features = out_features
        out_features = 90
        self.fc7 = nn.Linear(in_features, out_features)
        #self.dropout7 = nn.Dropout2d(0.00)

        in_features = out_features
        out_features = 90
        self.fc8 = nn.Linear(in_features, out_features)
        #self.dropout8 = nn.Dropout2d(0.00)

        in_features = out_features
        out_features = 48
        self.fc9 = nn.Linear(in_features, out_features)
        #self.dropout9 = nn.Dropout2d(0.00)

        in_features = out_features
        out_features = 24
        self.fc10 = nn.Linear(in_features, out_features)
        #self.dropout10 = nn.Dropout2d(0.00)

        in_features = out_features
        out_features = 12
        self.fc11 = nn.Linear(in_features, out_features)
        #self.dropout11 = nn.Dropout2d(0.00)

        in_features = out_features
        out_features = 12
        self.fc12 = nn.Linear(in_features, out_features)
        #self.dropout12 = nn.Dropout2d(0.00)

        in_features = out_features
        out_features = 12
        self.fc13 = nn.Linear(in_features, out_features)
        #self.dropout13 = nn.Dropout2d(0.00)

        in_features = out_features
        out_features = 5
        self.fc14 = nn.Linear(in_features, out_features)
        #self.dropout14 = nn.Dropout2d(0.00)

        in_features = out_features
        out_features = 5
        self.fc15 = nn.Linear(in_features, out_features)
        #self.dropout15 = nn.Dropout2d(0.00)

        in_features = out_features
        out_features = 5
        self.fc16 = nn.Linear(in_features, out_features)

        in_features = out_features
        out_features = classCount
        self.fc17 = nn.Linear(in_features, out_features)

    def forward(self, x):
        # sourcery skip: inline-immediately-returned-variable
        """
        Do forward pass.

        Args:
            x: represents one sample of the data.
        """
        # Pass data through ln1
        x = self.ln1(x)

        x = self.fc1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout4(x)

        x = self.fc5(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout7(x)

        x = self.fc8(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout8(x)

        x = self.fc9(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout9(x)

        x = self.fc10(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout10(x)

        x = self.fc11(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout11(x)

        x = self.fc12(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout12(x)

        x = self.fc13(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout13(x)

        x = self.fc14(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout14(x)

        x = self.fc15(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        #x = self.dropout15(x)

        x = self.fc16(x)
        x = F.relu(x)

        x = self.fc17(x)


        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
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
            Y_train
            ):
    """
    Train the model.

    Args:
        args: argparse.Namespace,
        model: nn.Module,
        optimizer
        X_train,
        Y_train

    Returns:
        history as a dictionary
            history['acc'] = accur
            history['losses'] = losses
    """
    global SAMBANOVA
    global DEEPHYPER

    numEpochs = args.epochs

    if SAMBANOVA or DEEPHYPER:
        if args.dry_run:
            args.niter = 1
            numEpochs = args.niter

    trainset = dataset(X_train, Y_train)
    #DataLoader
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False)

    loss_fn = nn.BCELoss()

    #forward loop
    losses = []
    accur = []
    for i in range(numEpochs):





        # TODO: This should be a batch.
        for _, (x_train, y_train) in enumerate( trainloader ):

            #calculate output
            output = model.forward(x_train)

            #calculate loss
            loss = loss_fn( output, y_train.reshape(-1,1) )

            #accuracy
            predicted = model( torch.tensor(x, dtype=torch.float32) )
            acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i%50 == 0:
            losses.append(loss)
            accur.append(acc)
            print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))

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


    timer.start('loading data')
    (x_train, y_train), (x_valid, y_valid) = load_data(config)
    timer.end('loading data')


    timer.start('preprocessing')
    model = Model1(config)

    # SambaNova conversions.
    #model.bfloat16().float()
    #samba.from_torch_(model)



    # BCELoss is correct for Model1.
    # Moved to train().
    #criterion = nn.BCELoss()

    # setup optimizer
    #optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizer = optim.AdamW(model.parameters())


    # patient early stopping
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)

    timer.end('preprocessing')

    #
    # Training
    #
    timer.start('model training')

    history = train(config, model, optimizer, x_train, y_train)

    timer.end('model training')

    HISTORY = history
    return history['acc'][-1]


if __name__ == '__main__':
    config = {
        'units': 10,
        'activation': 'relu',  # can be gelu
        'optimizer':  'Adam',  # can be AdamW but it has to be installed.
        'loss':       'binary_crossentropy',
        'batch_size': 4096,
        'epochs':     15,
        'dropout':    0.05,
        'patience':   12,
        'embed_hidden_size': 21,    # May not get used.
        'proportion': .80           # A value between [0., 1.] indicating how to split data between
                                    # training set and validation set. `prop` corresponds to the
                                    # ratio of data in training set. `1.-prop` corresponds to the
                                    # amount of data in validation set.
    }
    accuracy = run(config)
    print('accuracy: ', accuracy)

    import matplotlib.pyplot as plt
    plt.plot(HISTORY['acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
