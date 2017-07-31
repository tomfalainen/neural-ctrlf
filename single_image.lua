require 'torch'
require 'nn'
require 'cunn'
require 'dpnn'
require 'cudnn'

require 'ctrlfnet.DataLoader'
require 'ctrlfnet.WordSpottingModel'

utils = require 'ctrlfnet.utils'
box_utils = require 'ctrlfnet.box_utils'

cmd = torch.CmdLine()
cmd:option('-model_file', '', 'which model file to use')
cmd:option('-h5_file', '', 'which h5 file to use')
cmd:option('-gpu', 0, 'The GPU to use')
opt = cmd:parse(arg)
cutorch.setDevice(opt.gpu + 1) -- note +1 because lua is 1-indexed

-- function evaluate_single_image(model_file, img, region_proposals)
function evaluate_single_image(model_file, h5)
	checkpoint = torch.load(model_file)
	model = checkpoint.model
	model:evaluate()
	model.timing = false
	model.dump_vars = false
	model.cnn_backward = false
	opt = {rpn_nms_thresh=0.7, final_nms_thresh=-1}
	model:setTestArgs(opt)
    apply = nn.ApplyBoxTransform():cuda()
    
    h5_file = hdf5.open(h5, 'r')
    img = h5_file:read('/img'):all()
    region_proposals = h5_file:read('/region_proposals'):all()
    h5_file:close()
    
    im_size = img:size()
    rp_size = region_proposals[1]:size()
  	num_proposals = rp_size[1]
  	rpn_nms_thresh = 0.7
    batch_size = 1024
    embedding_size = 108
    
    img = img:cuda()
    region_proposals = region_proposals:cuda()
    	
    feature_maps = model.nets.conv_net2:forward(model.nets.conv_net1:forward(img))
	rpn_out = model.nets.localization_layer.nets.rpn:forward(feature_maps)
	rpn_boxes, rpn_anchors = rpn_out[1], rpn_out[2]
	rpn_trans, rpn_scores = rpn_out[3], rpn_out[4]
	num_boxes = rpn_boxes:size(2)

	bounds = {
	x_min=1, y_min=1,
	x_max=im_size[4], --image_width,
	y_max=im_size[3], --image_height
	}
	rpn_boxes, valid = box_utils.clip_boxes(rpn_boxes, bounds, 'xcycwh')

	function clamp_data(data)
	-- data should be 1 x kHW x D
	-- valid is byte of shape kHW
	assert(data:size(1) == 1, 'must have 1 image per batch')
	assert(data:dim() == 3)
	local mask = valid:view(1, -1, 1):expandAs(data)
	return data[mask]:view(1, -1, data:size(3))
	end
	rpn_boxes = clamp_data(rpn_boxes)
	rpn_scores = clamp_data(rpn_scores)
	num_boxes = rpn_boxes:size(2)

	rpn_boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(rpn_boxes)

	-- Convert objectness positive / negative scores to probabilities
	rpn_scores_exp = torch.exp(rpn_scores)
	pos_exp = rpn_scores_exp[{1, {}, 1}]
	neg_exp = rpn_scores_exp[{1, {}, 2}]
	scores = (pos_exp + neg_exp):pow(-1):cmul(pos_exp)

	boxes_scores = scores.new(num_boxes, 5)
	boxes_scores[{{}, {1, 4}}] = rpn_boxes_x1y1x2y2
	boxes_scores[{{}, 5}] = scores
	idx = box_utils.nms(boxes_scores, rpn_nms_thresh, num_proposals)

	rpn_boxes_nms = rpn_boxes:index(2, idx)[1]
	rpn_scores_nms = rpn_scores:index(2, idx)[1]
	
	-- Extract embeddings for rpn_boxes in batches
	boxes = torch.FloatTensor(num_proposals, 4)
	embeddings = torch.FloatTensor(num_proposals, embedding_size)
	logprobs = torch.FloatTensor(num_proposals, 1)
	box_split = boxes:split(batch_size)
	emb_split = embeddings:split(batch_size)
    lp_split = logprobs:split(batch_size)
	
	for iv,v in ipairs(rpn_boxes_nms:split(batch_size)) do
      model.nets.localization_layer.nets.roi_pooling:setImageSize(im_size[3], im_size[4])
      roi_features = model.nets.localization_layer.nets.roi_pooling:forward{feature_maps[1], v}
      roi_features = model.nets.recog_base:forward(roi_features)
      emb = model.nets.embedding_net:forward(roi_features)  
      lp = model.nets.objectness_branch:forward(roi_features)
      box_trans = model.nets.box_reg_branch:forward(roi_features)
      box = apply:forward({v, box_trans})

      emb_split[iv]:copy(emb:float())
      lp_split[iv]:copy(lp:float())
      box_split[iv]:copy(box:float())
    end
    boxes = box_utils.xcycwh_to_x1y1x2y2(boxes)
	
    -- Extract embeddings for external region proposals
    region_proposals = box_utils.x1y1x2y2_to_xcycwh(region_proposals)

    rp_size = region_proposals:size()
    rp_embeddings = torch.FloatTensor(rp_size[1], embedding_size)
    rp_scores = torch.FloatTensor(rp_size[1], 1)
    rpe_split = rp_embeddings:split(batch_size)
    rps_split = rp_scores:split(batch_size)

    for iv,v in ipairs(region_proposals:split(batch_size)) do
      model.nets.localization_layer.nets.roi_pooling:setImageSize(im_size[3], im_size[4])
      roi_features = model.nets.localization_layer.nets.roi_pooling:forward{feature_maps[1], v}
      roi_features = model.nets.recog_base:forward(roi_features)
      rpe = model.nets.embedding_net:forward(roi_features)  
      rps = model.nets.objectness_branch:forward(roi_features)

      rpe_split[iv]:copy(rpe:float())
      rps_split[iv]:copy(rps:float())
    end
    
    h5_file = hdf5.open(h5, 'w')
    h5_file:write('/boxes', boxes)
    h5_file:write('/embeddings', embeddings)
    h5_file:write('/logprobs', logprobs)
    h5_file:write('/rp_embeddings', rp_embeddings)
    h5_file:write('/rp_scores', rp_scores)
    h5_file:close()
    
    -- return {boxes, embeddings, logprobs, rp_embeddings, rp_scores}
end


evaluate_single_image(opt.model_file, opt.h5_file)