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


SAMBANOVA = True
DEEPHYPER = False
DEVICE    = None
DTYPE     = None

if SAMBANOVA:
    from sambaflow import samba
    import sambaflow.samba.utils as utils
    from sambaflow.samba.utils.argparser import parse_app_args
    from sambaflow.samba.utils.pef_utils import get_pefmeta

    if DEEPHYPER:
        from model1.model1.m1_hps.model_args_pytorch import *
    else:
        from model_args_pytorch import *

class Model1(nn.Module):
    """Model 1 object."""

    # dtype=float != torch.float

    def __init__(self, config):
        """Initialize the model object."""
        super(Model1, self).__init__()

        samples = 65
        batchSamples = 26

        numInputs = samples * batchSamples
        input_shape = (5278, numInputs)
        #classCount = self.getClassCount()

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





        #"input_layers": [["input_0", 0, 0]],
        self.layer_norm = nn.LayerNorm(input_shape).to(device, dtype=dtype)

        #if batchSamples >= 16:
        in_features = numInputs
        out_features = 80
        self.dense = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout_1 = nn.Dropout2d(dropout1).to(device, dtype=dtype)

        self.activation = torch.nn.Sigmoid().to(device, dtype=dtype)


        #if batchSamples >= 12:
        in_features = out_features
        out_features = 80
        self.dense_1 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout_2 = nn.Dropout2d(dropout2).to(device, dtype=dtype)


        out_features = 5278
        self.activation_1 = torch.nn.ReLU().to(device, dtype=dtype)

        #if batchSamples >= 10:
        in_features = out_features
        out_features = 80
        self.dense_2 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout_3 = nn.Dropout2d(dropout2).to(device, dtype=dtype)

        out_features = 5278
        self.activation_2 = torch.nn.Tanh().to(device, dtype=dtype)

        in_features = out_features
        out_features = 80
        self.dense_3 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout_4 = nn.Dropout2d(dropout3).to(device, dtype=dtype)

        out_features = 80
        self.activation_3 = torch.nn.ReLU().to(device, dtype=dtype)

        in_features = out_features
        out_features = 80
        self.dense_4 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout_5 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        out_features = 5278
        self.activation_4 = torch.nn.Sigmoid().to(device, dtype=dtype)

        in_features = out_features
        out_features = 80
        self.dense_5 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout_6 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        in_features = out_features
        out_features = 80
        self.dense_6 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        self.dropout_7 = nn.Dropout2d(dropout4).to(device, dtype=dtype)

        out_features = 5278
        self.activation_5 = torch.nn.ReLU().to(device, dtype=dtype)


        in_features = out_features
        out_features = 2 # This is 2 instead of 1 due to the PyTorch softmax.
        self.dense_7 = nn.Linear(in_features, out_features).to(device, dtype=dtype)
        #"output_layers": [["dense_7", 0, 0]]}


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # sourcery skip: inline-immediately-returned-variable
        """
        Do forward pass.

        Args:
            x: represents one sample of the data.
        """
        # Does this need to use targets to calculate loss?


        device = DEVICE
        dtype  = DTYPE

        print(f"inputs.shape = {inputs.shape}") # torch.Size([5278, 1690])
        input_0 = inputs
        input_0 = self.layer_norm(input_0)  # (1, 1690)
        dense = self.dense(input_0)
        dense = self.dropout_1(dense)
        activation = self.activation(dense)
        input_0 = activation                # (1, 80)
        print(f"activation.shape = {activation.shape}")

        dense_1 = self.dense_1(input_0)
        dense_1 = self.dropout_2(dense_1)
        add = activation + dense_1

        # Can use ReLU(inplace=False)
        activation_1 = self.activation_1(add)
        print(f"activation_1.shape: {activation_1.shape}")

        activation_1_t = torch.transpose(activation_1, 0, 1)
        dense_2 = self.dense_2( activation_1_t )  # It is an input * weight problem.  No.
        dense_2 = self.dropout_3(dense_2)
        activation_2 = self.activation_2(dense_2)
        print(f"activation_2.shape: {activation_2.shape}")

        dense_3 = self.dense_3(activation_1_t)
        dense_3 = self.dropout_4(dense_3)
        add_1 = activation_2 + dense_3
        # Can use ReLU(inplace=False)
        activation_3 = self.activation_3(add_1)
        print(f"activation_3.shape: {activation_3.shape}")

        activation_3_t = torch.transpose(activation_3, 0, 1)
        dense_4 = self.dense_4(activation_3_t)
        dense_4 = self.dropout_5(dense_4)
        activation_4 = self.activation_4(dense_4)

        input_0_t = torch.transpose(input_0, 0, 1)
        dense_5 = self.dense_5(input_0_t)
        dense_5 = self.dropout_6(dense_5)

        dense_6 = self.dense_6(activation_3_t)
        dense_6 = self.dropout_7(dense_6)

        dense_5_t = torch.transpose(dense_5, 0, 1)
        dense_6_t = torch.transpose(dense_5, 0, 1)

        print(f"activation_4.shape: {activation_4.shape}")
        print(f"dense_5.shape: {dense_5.shape}")
        print(f"dense_6.shape: {dense_6.shape}")
        add_2 = activation_4 + dense_5 + dense_6
        # Can use ReLU(inplace=False)
        activation_5 = self.activation_5(add_2)

        activation_5_t = torch.transpose(activation_5, 0, 1)
        dense_7 = self.dense_7(activation_5_t)

        # Apply log_softmax to x
        output = F.softmax(dense_7, dim=1)
        return output

    @staticmethod
    def get_fake_inputs(args):
        """
        Get fake inputs.

        The size of the inputs are required for the SambaNova compiler.

        Args:
            args: CLI arguments.
        """
        ipt = samba.randn(args.batch_size, args.num_features, name='image', batch_dim=0).bfloat16().float()
        tgt = samba.randint(args.num_classes, (args.batch_size, ), name='label', batch_dim=0)

        return ipt, tgt

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


def to_torch_tensor(x_train):
    # sourcery skip: inline-immediately-returned-variable
    """Convert np.array to torch tensor."""
    arrayOf2dList = x_train
    numpyArrayOf2dListFloat64 = np.array( arrayOf2dList )
    numpyArrayOf2dListFloat32 = numpyArrayOf2dListFloat64.astype(np.float32)
    torch_tensor = torch.from_numpy( numpyArrayOf2dListFloat32 )
    return torch_tensor

def run(config, argv):
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
        #(x_train, y_train), (x_valid, y_valid) = load_data(config)
        #####timer.end('loading data')


        #####timer.start('preprocessing')

        if SAMBANOVA:
            """Run main code."""
            utils.set_seed(256)
            args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"*** device: {device} ***")

        dtype = torch.float if device == "cuda" else torch.float32

        config['device'] = device
        config['dtype']  = dtype

        DEVICE = device
        DTYPE  = dtype

        if SAMBANOVA:
            ipt, tgt = Model1.get_fake_inputs(args)

        model = Model1(config).to(device, dtype=dtype)

        if SAMBANOVA:
            #x_train = to_torch_tensor(x_train)
            #y_train = to_torch_tensor(y_train)

            from torchsummary import summary

            #input_data  = x_train
            #targets     = y_train
            #input_shape  = (1, x_train.shape[1])
            #print(f"input_shape: {input_shape}")

            """
            model_stats = summary(model, input_shape)
            #model_stats = summary(model, input_data, targets)
            summary_str = str(model_stats)
            print(f"summary_str = {summary_str}")
            # summary_str contains the string representation of the summary. See below for examples.
            #https://pypi.org/project/torch-summary/
            """

            # SambaNova conversions.
            model.bfloat16().float()
            samba.from_torch_(model)
            inputs = (ipt, tgt)



        # BCELoss is correct for Model1.
        # Moved to train().
        #criterion = nn.BCELoss()

        # setup optimizer
        #optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
        optimizer = config['optimizer'].lower()

        if SAMBANOVA:
            args = args

            # Instantiate an optimizer.
            if args.inference:
                optimizer = None
            else:
                optimizer = samba.optim.SGD(model.parameters(),
                                            lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

            if args.command == "compile":
                # Run model analysis and compile, this step will produce a PEF.
                samba.session.compile(model,
                                    inputs,
                                    optimizer,
                                    name='model_1_torch',
                                    app_dir=utils.get_file_dir(__file__),
                                    config_dict=vars(args),
                                    pef_metadata=get_pefmeta(args, model))
            elif args.command == "test":
                utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
                outputs = model.output_tensors
                test(args, model, inputs, outputs)
            elif args.command == "run":
                utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
                train(args, model, optimizer)

        else:
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

    import sys
    accuracy = run(config, sys.argv[1:])
    print('accuracy: ', accuracy)

    """
    import matplotlib.pyplot as plt
    plt.plot(HISTORY['acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    """
