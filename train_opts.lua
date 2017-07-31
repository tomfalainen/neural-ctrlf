local M = { }

function M.parse(arg)

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a Ctrl-F-Net model.')
  cmd:text()
  cmd:text('Options')

  -- Core ConvNet settings
  cmd:option('-backend', 'cudnn', 'nn|cudnn')
  
  -- Model settings
  cmd:option('-rpn_hidden_dim', 128, 'Hidden size for the extra convolution in the RPN')
  cmd:option('-sampler_batch_size', 256, 'Batch size to use in the box sampler')
  cmd:option('-rnn_size', 512, 'Number of units to use at each layer of the RNN')
  cmd:option('-input_encoding_size', 512, 'Dimension of the word vectors to use in the RNN')
  cmd:option('-sampler_high_thresh', 0.75, 'Boxes with IoU greater than this with a GT box are considered positives')
  cmd:option('-sampler_low_thresh', 0.4, 'Boxes with IoU less than this with all GT boxes are considered negatives')
  cmd:option('-train_remove_outbounds_boxes', 1, 'Whether to ignore out-of-bounds boxes for sampling at training time')
  cmd:option('-embedding', 'dct', 'Which embedding to use, dct or phoc')
  cmd:option('-biased_sampling', 1, 'Whether or not to try to bias sampling to use uncommon words as often as possible.')
  
  -- Loss function weights
  cmd:option('-mid_box_reg_weight', 0.01,'Weight for box regression in the RPN')
  cmd:option('-mid_objectness_weight', 0.01, 'Weight for box classification in the RPN')
  cmd:option('-end_box_reg_weight', 0.1, 'Weight for box regression in the recognition network')
  cmd:option('-end_objectness_weight', 0.1, 'Weight for box classification in the recognition network')
  cmd:option('-embedding_weight',3.0, 'Weight for embedding loss')
  cmd:option('-weight_decay', 1e-5, 'L2 weight decay penalty strength')
  cmd:option('-box_reg_decay', 5e-5, 'Strength of pull that boxes experience towards their anchor')
  cmd:option('-cosine_margin', 0.1, 'margin for the cosine loss')

  -- Data input settings
  cmd:option('-dataset', '', 'HDF5 file containing the preprocessed dataset (from proprocess.py)')
  cmd:option('-val_dataset', 'washington', 'HDF5 file containing the preprocessed dataset (from proprocess.py)')
  cmd:option('-fold', 1, 'which fold to use')
  cmd:option('-dataset_path', 'data/dbs/', 'HDF5 file containing the preprocessed dataset (from proprocess.py)')
  cmd:option('-debug_max_train_images', -1,'Use this many training images (for debugging); -1 to use all images')

  -- Optimization
  cmd:option('-learning_rate', 2e-3, 'learning rate to use')
  cmd:option('-reduce_lr_every', 10000, 'reduce learning rate every x iterations')
  cmd:option('-optim_beta1', 0.9, 'beta1 for adam')
  cmd:option('-optim_beta2', 0.999, 'beta2 for adam')
  cmd:option('-optim_epsilon', 1e-8, 'epsilon for smoothing')
  cmd:option('-drop_prob', 0.5, 'Dropout strength throughout the model.')
  cmd:option('-max_iters', 25000, 'Number of iterations to run; -1 to run forever')
  cmd:option('-checkpoint_start_from', '', 'Load model from a checkpoint instead of random initialization.')
  cmd:option('-finetune_cnn_after', 1000, 'Start finetuning CNN after this many iterations (-1 = never finetune)')
  cmd:option('-val_images_use', -1, 'Number of validation images to use for evaluation; -1 to use all')

  -- Model checkpointing
  cmd:option('-save_checkpoint_every', 1000, 'How often to save model checkpoints')
  cmd:option('-checkpoint_path', 'checkpoints/','path to where checkpoints are saved')
  cmd:option('-checkpoint_start_from', '','Name of the checkpoint file to use')
    
  -- Test-time model options (for evaluation)
  cmd:option('-test_rpn_nms_thresh', 0.7, 'Test-time NMS threshold to use in the RPN')
  cmd:option('-test_final_nms_thresh', -1, 'Test-time NMS threshold to use for final outputs')
  cmd:option('-test_num_proposals', 1000, 'Number of region proposal to use at test-time')

  -- Visualization
  cmd:option('-print_every', 200, 'How often to print the latest images training loss.')
  cmd:option('-progress_dump_every', 100, 'After how many iterations do we write a progress report to vis/out ?. 0 = disable.')
  cmd:option('-losses_log_every', 10, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

  -- Misc
  cmd:option('-id', 'presnet', 'an id identifying this run/job; useful for cross-validation')
  cmd:option('-seed', 123, 'random number generator seed to use')
  cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU')
  cmd:option('-timing', false, 'whether to time parts of the net')
  cmd:option('-clip_final_boxes', 1, 'Whether to clip final boxes to image boundar')
  
  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

return M
