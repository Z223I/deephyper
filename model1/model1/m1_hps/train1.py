

def train(  args: argparse.Namespace,
            model: nn.Module,
            optimizerD: optim.SGD,
            optimizerG: optim.SGD) -> None:
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
    portWork = True
    if portWork:
        dataloader = model.cache["dataloader"]

        criterion = nn.BCELoss()

        batch_size  = model.cache["batch_size"]
        nz          = model.cache["nz"]
        device      = model.cache["device"]

        fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
        real_label = 1
        fake_label = 0

        netD = model.netD
        netG = model.netG

        for epoch in range(args.niter):
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
                    % (epoch, args.niter, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/real_samples.png' % args.outf,
                            normalize=True)
                    fake = netG(fixed_noise)
                    vutils.save_image(fake.detach(),
                            '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                            normalize=True)

                if args.dry_run:
                    break

            # do checkpointing
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))



    else:
        train_loader, test_loader = prepare_dataloader(args)

        # Train the model
        total_step = len(train_loader)
        hyperparam_dict = {"lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay}
        for epoch in range(args.num_epochs):
            avg_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                sn_images = samba.from_torch(images, name='image', batch_dim=0)
                sn_labels = samba.from_torch(labels, name='label', batch_dim=0)

                loss, outputs = samba.session.run(input_tensors=[sn_images, sn_labels],
                                                output_tensors=model.output_tensors,
                                                hyperparam_dict=hyperparam_dict,
                                                data_parallel=args.data_parallel,
                                                reduce_on_rdu=args.reduce_on_rdu)
                loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
                avg_loss += loss.mean()

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1, total_step,
                                                                        avg_loss / (i + 1)))

            samba.session.to_cpu(model)
            test_acc = 0.0
            with torch.no_grad():
                correct = 0
                total = 0
                total_loss = 0
                for images, labels in test_loader:
                    loss, outputs = model(images, labels)
                    loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
                    total_loss += loss.mean()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                test_acc = 100.0 * correct / total
                print('Test Accuracy: {:.2f}'.format(test_acc),
                    ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))

            if args.acc_test:
                assert args.num_epochs == 1, "Accuracy test only supported for 1 epoch"
                assert test_acc > 92.0 and test_acc < 94.0, "Test accuracy not within specified bounds."

        if args.json is not None:
            report = AccuracyReport(val_accuracy=test_acc.item(),
                                    batch_size=args.batch_size,
                                    num_iterations=args.num_epochs * total_step)
            report.save(args.json)
