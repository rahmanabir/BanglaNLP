for epoch in range(num_epochs):
    
    trainiter = iter(trainloader)
    
    for i in range(total_step-1):
        
        spectros, lbls, lbl_lens = trainiter.next()
        
#         print(lbls)
        
        spectros = spectros.to(device)
        
        
        
        lbls = lbls.to(device)
        lbl_lens.to(device)
        
        pred = model(spectros)
        
        print(pred.shape)
        print(spectros.shape)
        
        preds_size = Variable(torch.LongTensor([pred.size(0)] * batch_size))
        
        print(preds_size)
        print(lbls.shape)
        print(lbls[0])
        
        break
    
    break
#         #backprop and optimize!
        
        
#         optimizer.zero_grad()
#         cost = criterion(pred, lbls, preds_size, lbl_lens)/batch_size

#         cost.backward()
#         optimizer.step()