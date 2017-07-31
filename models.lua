local M = {}

function M.setup(opt)
  local model
  if opt.checkpoint_start_from == '' then
    print('initializing a DenseCap model from scratch...')
    model = WordSpottingModel(opt)
  else
    print('initializing a DenseCap model from ' .. opt.checkpoint_start_from)
    model = torch.load(opt.checkpoint_start_from).model
    model.opt.end_objectness_weight = opt.end_objectness_weight
    model.nets.localization_layer.opt.mid_objectness_weight = opt.mid_objectness_weight
    model.nets.localization_layer.opt.mid_box_reg_weight = opt.mid_box_reg_weight
    model.crits.box_reg_crit.w = opt.end_box_reg_weight
    local rpn = model.nets.localization_layer.nets.rpn
    rpn:findModules('nn.RegularizeLayer')[1].w = opt.box_reg_decay
    model.opt.train_remove_outbounds_boxes = opt.train_remove_outbounds_boxes
    model.opt.embedding_weight = opt.embedding_weight

    -- TODO: Move into a reset function in BoxSampler
    model.nets.localization_layer.nets.box_sampler_helper.box_sampler.vocab_size = opt.vocab_size
    model.nets.localization_layer.nets.box_sampler_helper.box_sampler.histogram = torch.ones(opt.vocab_size)

  end

  -- Find all Dropout layers and set their probabilities
  local dropout_modules = model.nets.recog_base:findModules('nn.Dropout')
  for i, dropout_module in ipairs(dropout_modules) do
    dropout_module.p = opt.drop_prob
  end

  return model
end

return M
