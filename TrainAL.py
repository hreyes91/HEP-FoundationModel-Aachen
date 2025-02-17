







    original_model = torch.load(os.path.join(args.model_path_in, args.model_name))
    # construct model
    model = JetTransformerAL(original_model,
        #hidden_dim=args.hidden_dim,
        original_model,
        hidden_dim=256,
        num_layers=10,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        num_const=100
        
        )
    model.to(device)
    path_to_sate_dict = os.path.join(args.model_path_in, 'opt_state_dict_best.pt')

    filtered_opt_state_dict=orig_load_opt_dict(args.model_path_in,path_to_sate_dict)
    # construct optimizer and auto-caster
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    #filtered_opt_state_dict=UpdateOpt(filtered_opt_state_dict,opt,model)
    print('model paramaters')
    print(model.parameters())
    print('opt state dict')
    print(opt.state_dict())
    
    last2paramgroups, last2state,last_keys=GetLast2Layers(opt.state_dict())
    filtered_opt_state_dict= AddLayersToDict(filtered_opt_state_dict,last2state,last2paramgroups,last_keys)
    opt.load_state_dict(filtered_opt_state_dict)
    
    print(opt.state_dict())
    scheduler = get_cos_scheduler(
        num_epochs=args.num_epochs,
        num_batches=len(train_loader),
        optimizer=opt,
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    
######################################################################
    logger = SummaryWriter(args.log_dir)
    global_step = 0
    loss_list = []
    loss_list_epoch=[]
    val_list_epoch=[]
    
    
    perplexity_list = []
    min_val_loss = np.inf
    for epoch in range(args.num_epochs):
        model.train()
        loss_list_here=[]
        for x, padding_mask, label in tqdm(
            train_loader, total=len(train_loader), desc=f"Training Epoch {epoch + 1}"
        ):
            opt.zero_grad()
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            label = label.to(device)

            with torch.cuda.amp.autocast():
                logits = model(x, padding_mask)
                loss = model.loss(logits, label.view(-1, 1))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            #print('losss')
            #print(loss.cpu().detach().numpy())
            #print(float(loss.cpu().detach().numpy()))
            loss_list.append(loss.cpu().detach().numpy())
            loss_list_here.append(loss.cpu().detach().numpy())
            if (global_step + 1) % args.logging_steps == 0:
                logger.add_scalar("Train/Loss", np.mean(loss_list), global_step)
                logger.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
                loss_list = []
                perplexity_list = []

            global_step += 1

        model.eval()
        with torch.no_grad():
            val_loss = []
            val_perplexity = []
            for x, padding_mask, label in tqdm(
                val_loader, total=len(val_loader), desc=f"Validation Epoch {epoch + 1}"
            ):
                x = x.to(device)
                padding_mask = padding_mask.to(device)
                label = label.to(device)

                logits = model(
                    x,
                    padding_mask,
                )
                loss = model.loss(logits, label.view(-1, 1))
                val_loss.append(loss.cpu().detach().numpy())
                val_loss_here=val_loss
            val_loss = np.mean(val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_model(model, args.log_dir, "best")
            logger.add_scalar("Val/Loss", np.mean(val_loss), global_step)
        
        save_model(model, args.log_dir, "last")
        save_opt_states(
            optimizer=opt, scheduler=scheduler, scaler=scaler, log_dir=args.log_dir
        )
        mean_loss=np.mean(loss_list)
        mean_val=val_loss
        loss_list_epoch.extend(loss_list_here)
        val_list_epoch.extend(val_loss_here)
    print(loss_list_epoch)
    print(len(loss_list_epoch))
    print(val_list_epoch)
    print(len(val_list_epoch))
    
    
    history={'loss':loss_list_epoch,'val_loss':val_list_epoch}
    
    history_frame=pd.DataFrame(history)
    history_frame.to_csv(os.path.join(args.log_dir, "history.txt"),index=False)
    
    
    
    plot_rocs(model, val_loader, tag="last")
    model = load_model(os.path.join(args.log_dir, "model_best.pt"))
    plot_rocs(model, val_loader, tag="best")

plt.close()
plt.close()
import matplotlib.pyplot as plt
plt.plot(history_frame['loss'], label='Train Loss')
plt.plot(history_frame['val_loss'], label='Val Loss')
plt.xlabel('iter')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig(os.path.join(args.log_dir, "history.pdf"))
plt.close()

