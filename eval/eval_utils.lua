local cjson = require 'cjson'
local utils = require 'ctrlfnet.utils'
local box_utils = require 'ctrlfnet.box_utils'

local eval_utils = {}

--[[
Evaluate a DenseCapModel on a split of data from a DataLoader.

Input: An object with the following keys:
- model: A DenseCapModel object to evaluate; required.
- loader: A DataLoader object; required.
- split: Either 'val' or 'test'; default is 'val'
- max_images: Integer giving the number of images to use, or -1 to use the
  entire split. Default is -1.
- id: ID for cross-validation; default is ''.
- dtype: torch datatype to which data should be cast before passing to the
  model. Default is 'torch.FloatTensor'.
--]]
function eval_utils.eval_split(kwargs)
  local model = utils.getopt(kwargs, 'model')
  local loader = utils.getopt(kwargs, 'loader')
  local fold = utils.getopt(kwargs, 'fold')
  local val_dataset = utils.getopt(kwargs, 'val_dataset')
  local split = utils.getopt(kwargs, 'split', 'val')
  local max_images = utils.getopt(kwargs, 'max_images', -1)
  local embedding = utils.getopt(kwargs, 'embedding')
  local id = utils.getopt(kwargs, 'id', '')
  local dtype = utils.getopt(kwargs, 'dtype', 'torch.FloatTensor')
  assert(split == 'val' or split == 'test', 'split must be "val" or "test"')
  local split_to_int = {val=1, test=2}
  split = split_to_int[split]
  -- print('using split ', split)
  
  model:evaluate()
  model:clearState()
  local embedding_size = 108
  if embedding == 'phoc' then
    embedding_size = 540
  elseif embedding == 'bigram_phoc' then
    embedding_size = 604
  end

  loader:resetIterator(split)

  local all_losses = {}
  local all_boxes = {}
  local all_logprobs = {}
  local all_embeddings = {}
  local all_gt_embeddings = {}
  local all_gt_scores = {}
  local all_gt_boxes = {}
  local all_rp_embeddings = {}
  local all_rp_scores = {}
  local all_region_proposals = {}
  local btoi = {}
  local rptoi = {}

  local j = 1 -- never, EVER change to 0 instead of 1 for j & k
  local k = 1

  local counter = 0
  local all_losses = {}
  while true do
    counter = counter + 1
    
    -- Grab a batch of data and convert it to the right dtype
    local data = {}
    local loader_kwargs = {split=split, iterate=true}
    local img, gt_boxes, region_proposals, gt_embeddings, labels, info = loader:getBatch(loader_kwargs)

    local data = {
      image = img:type(dtype),
      gt_boxes = gt_boxes:type(dtype),
      embeddings = gt_embeddings:type(dtype),
      labels = labels:type(dtype)
    }

    local im_size = #data.image

    local boxes, logprobs, embeddings = model:forward_test(data.image)
    
    boxes = box_utils.xcycwh_to_x1y1x2y2(boxes)
    table.insert(all_boxes, boxes:float())
    table.insert(all_logprobs, logprobs:float())
    table.insert(all_embeddings, embeddings:float())

    local feature_maps = model.nets.conv_net2:forward(model.nets.conv_net1:forward(data.image))
    gt_boxes = gt_boxes[1]:cuda()

    model.nets.localization_layer.nets.roi_pooling:setImageSize(im_size[3], im_size[4])
    local roi_features = model.nets.localization_layer.nets.roi_pooling:forward{feature_maps[1], gt_boxes}
    roi_features = model.nets.recog_base:forward(roi_features)
    local qbe_gt_embeddings = model.nets.embedding_net:forward(roi_features) 
    local gt_scores = model.nets.objectness_branch:forward(roi_features)


    gt_boxes =  box_utils.xcycwh_to_x1y1x2y2(gt_boxes)
    table.insert(all_gt_embeddings, qbe_gt_embeddings:float())
    table.insert(all_gt_boxes, gt_boxes:float())
    table.insert(all_gt_scores, gt_scores:float())

    -- Extract embeddings for external region proposals
    region_proposals = region_proposals[1]:cuda()

    local batch_size = 1024
    local rp_size = region_proposals:size()
    local rp_embeddings = torch.FloatTensor(rp_size[1], embedding_size)
    local rp_scores = torch.FloatTensor(rp_size[1], 1)
    local rpe_split = rp_embeddings:split(batch_size)
    local rps_split = rp_scores:split(batch_size)

    for iv,v in ipairs(region_proposals:split(batch_size)) do
      model.nets.localization_layer.nets.roi_pooling:setImageSize(im_size[3], im_size[4])
      roi_features = model.nets.localization_layer.nets.roi_pooling:forward{feature_maps[1], v}
      roi_features = model.nets.recog_base:forward(roi_features)
      local rpe = model.nets.embedding_net:forward(roi_features)  
      local rps = model.nets.objectness_branch:forward(roi_features)
      
      rpe_split[iv]:copy(rpe:float())
      rps_split[iv]:copy(rps:float())
    end

    region_proposals = box_utils.xcycwh_to_x1y1x2y2(region_proposals:float())
    table.insert(all_region_proposals, region_proposals)
    table.insert(all_rp_scores, rp_scores)
    table.insert(all_rp_embeddings, rp_embeddings)

    -- This needs to be last as calling model:forward_backward effects the results of the embeddings extracted, not sure how yet. 
    -- Call forward_backward to compute losses
    model.timing = false
    model.dump_vars = false
    model.cnn_backward = false
    local losses = model:forward_backward(data)
    table.insert(all_losses, losses)

    for i = j, (j + boxes:size(1) - 1) do
      btoi[i] = counter
    end
    j = j + boxes:size(1)

    for i = k, (k + rp_size[1] - 1) do
      rptoi[i] = counter
    end
    k = k + rp_size[1]
    
    -- Print a message to the console
    local msg = 'Processed image %s (%d / %d) of split %d, detected %d regions'
    local num_images = info.split_bounds[2]
    if max_images > 0 then num_images = math.min(num_images, max_images) end
    local num_boxes = boxes:size(1)
    if counter % 20 == 0 then
      print(string.format(msg, info.filename, counter, num_images, split, num_boxes))
    end

    -- Break out if we have processed enough images
    if max_images > 0 and counter >= max_images then break end
    if info.split_bounds[1] == info.split_bounds[2] then break end

  end

  local loss_results = utils.dict_average(all_losses)
  print('Validation Average loss: ', loss_results)
  
  local net = nn.JoinTable(1)

  h5_file = hdf5.open('tmp/' .. val_dataset .. '_' .. embedding .. '_descriptors.h5', 'w')

    -- Avoiding deep copies, which is otherwise needed
  local l = net:forward(all_logprobs)
  h5_file:write('/logprobs_fold' .. fold, l)
  local e = net:forward(all_embeddings)
  h5_file:write('/embeddings_fold' .. fold, e)
  local b = net:forward(all_boxes)
  h5_file:write('/boxes_fold' .. fold, b)
  local gte = net:forward(all_gt_embeddings)
  h5_file:write('/gt_embeddings_fold' .. fold, gte)
  local gts = net:forward(all_gt_scores)
  h5_file:write('/gt_scores_fold' .. fold, gts)
  local rps = net:forward(all_rp_scores)
  h5_file:write('/rp_scores_fold' .. fold, rps)
  local rp = net:forward(all_region_proposals)
  h5_file:write('/region_proposals_fold' .. fold, rp)
  local rpe = net:forward(all_rp_embeddings)
  h5_file:write('/rp_embeddings_fold' .. fold, rpe)
  local gtb = net:forward(all_gt_boxes)
  h5_file:write('/gt_boxes_fold' .. fold, gtb)
  local btoi = torch.FloatTensor(btoi)
  h5_file:write('/box_to_images_fold' .. fold, btoi)
  local rptoi = torch.FloatTensor(rptoi)
  h5_file:write('/rp_to_images_fold' .. fold, rptoi)

  h5_file:close()

  local ap_results = utils.map_eval(val_dataset, id, embedding, fold)
  print(string.format('QbE mAP: %f%%, QbE recall: %f%%, QbS mAP: %f%%, QbS recall: %f%%, total_recall: %f%%, rpn_recall: %f%%', 
    100 * ap_results.mAP_qbe, 100 * ap_results.recall_qbe, 100 * ap_results.mAP_qbs, 100 * ap_results.recall_qbs, 100 * ap_results.total_recall, 100*ap_results.rpn_recall))
  
  local out = {
    loss_results=loss_results,
    ap_results=ap_results,
  }
  return out
end

return eval_utils
