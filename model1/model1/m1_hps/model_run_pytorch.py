"""PyTorch version of Model 1."""

from deephyper.search.util import Timer
timer = Timer()
timer.start("module loading")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import sambaflow.samba.optim as optim
#import EarlyStopping

"""
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def validation_step(...):
    self.log('val_loss', loss)

trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss')])
"""
from model1.model1.m1_hps.load_data_pytorch import load_data

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

        in_features = numInputs
        out_features = samples * 16
        self.fc1 = nn.Linear(in_features, out_features)
        self.dropout1 = nn.Dropout2d(0.20)

        in_features = out_features
        out_features = samples * 12
        self.fc2 = nn.Linear(in_features, out_features)
        self.dropout2 = nn.Dropout2d(0.10)



        # TODO: Insert more layers here.



        in_features = out_features
        out_features = 5
        self.fcN_1 = nn.Linear(in_features, out_features)

        in_features = out_features
        out_features = classCount
        self.fcN = nn.Linear(in_features, out_features)

    def forward(self, x):
        # sourcery skip: inline-immediately-returned-variable
        """
        Do forward pass.

        Args:
            x: represents our data.
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


        # TODO: Insert more layers here.


        x = self.fcN_1(x)
        x = F.relu(x)

        x = self.fcN(x)


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
            ) -> None:
    """
    Train the model.

    Args:
        args: argparse.Namespace,
        model: nn.Module,
        optimizerD: optim.SGD,
        optimizerG: optim.SGD

    Returns:
        None
    """
    global SAMBANOVA
    global DEEPHYPER

    if SAMBANOVA or DEEPHYPER:
        if args.dry_run:
            args.niter = 1
            numEpochs = args.niter

    """
    for epoch in range(numEpochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                            dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, opt.niter, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)

            if opt.dry_run:
                break
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    """


"""
# This code is from a pytorch-lightning early stopping example.
#Train the Model using Early Stopping
def train_model(model, batch_size, patience, n_epochs):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    # https://github.com/Bjarten/early-stopping-pytorch
    #early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for batch, (data, target) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

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
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses
"""



# Track model history.
HISTORY = None


def run(config):
    """Run model."""
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

    timer.start('model training')

    # evaluate the model
    # TODO:  Why is loss used here.
    #loss, accuracy, f1_m, precision_m, recall_m = modelAnalyzeThis.evaluate(Xtest, Ytest, verbose=0)
    #acc = model.evaluate(Xtest, yTest, verbose=0)

    timer.end('model training')

    #HISTORY = history.history

    accJunk = 0.0
    return accJunk
    #return history.history['acc'][-1]


if __name__ == '__main__':
    config = {
        'units': 10,
        'activation': 'relu',  # can be gelu
        'optimizer':  'Adam',  # can be AdamW but it has to be installed.
        'loss':       'binary_crossentropy',
        'batch_size': 4096,
        'epochs':     200,
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