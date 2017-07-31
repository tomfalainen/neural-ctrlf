require 'hdf5'
local utils = require 'ctrlfnet.utils'
local box_utils = require 'ctrlfnet.box_utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  self.dataset = utils.getopt(opt, 'dataset')
  self.dataset_path = utils.getopt(opt, 'dataset_path')
  self.debug_max_train_images = utils.getopt(opt, 'debug_max_train_images', -1)

  self.h5_file = self.dataset_path .. self.dataset .. '.h5'
  self.json_file = self.dataset_path .. self.dataset .. '.json'
  
  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', self.json_file)
  self.info = utils.read_json(self.json_file)
  self.vocab_size = utils.count_keys(self.info.itow)

  -- Convert keys in idx_to_token from string to integer
  local itow = {}
  for k, v in pairs(self.info.itow) do
    itow[tonumber(k)] = v
  end
  self.info.itow = itow

  -- open the hdf5 file
  print('DataLoader loading h5 file: ', self.h5_file)
  self.h5_file = hdf5.open(self.h5_file, 'r')
  local keys = {}
  -- table.insert(keys, 'box_to_img')
  table.insert(keys, 'boxes')
  table.insert(keys, 'image_heights')
  table.insert(keys, 'image_widths')
  table.insert(keys, 'img_to_first_box')
  table.insert(keys, 'img_to_last_box')
  table.insert(keys, 'labels')
  table.insert(keys, opt.embedding .. '_word_embeddings')
  table.insert(keys, 'img_to_first_rp')
  table.insert(keys, 'img_to_last_rp')
  table.insert(keys, 'original_heights')
  table.insert(keys, 'original_widths')
  table.insert(keys, 'split')
  for k,v in pairs(keys) do
    -- print('reading ' .. v)
    self[v] = self.h5_file:read('/' .. v):all()
  end
 
  self.embedding = opt.embedding
  self.ordered = utils.ArrangeByLabel(self.labels) -- Used for sampling pairs for embedding loss.

  -- extract image size from dataset
  local images_size = self.h5_file:read('/images'):dataspaceSize()
  assert(#images_size == 4, '/images should be a 4D tensor')
  self.num_images = images_size[1]
  self.num_channels = images_size[2]
  self.max_image_height = images_size[3]
  self.max_image_width = images_size[4]

  -- extract some attributes from the data
  self.num_regions = self.boxes:size(1)
  self.mean = self.h5_file:read('/image_mean'):all()[1]

  -- set up index ranges for the different splits
  self.train_ix = {}
  self.val_ix = {}
  self.test_ix = {}
  for i=1,self.num_images do
    if self.split[i] == 0 then table.insert(self.train_ix, i) end
    if self.split[i] == 1 then table.insert(self.val_ix, i) end
    if self.split[i] == 2 then table.insert(self.test_ix, i) end
  end

  self.iterators = {[0]=1,[1]=1,[2]=1} -- iterators (indices to split lists) for train/val/test
  print(string.format('assigned %d/%d/%d images to train/val/test.', #self.train_ix, #self.val_ix, #self.test_ix))

  print('initialized DataLoader:')
  print(string.format('#images: %d, #regions: %d', self.num_images, self.num_regions))
end

function DataLoader:getImageMaxSize()
  return self.max_image_height, max_image_width
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.info.itow
end

-- split is an integer: 0 = train, 1 = val, 2 = test
function DataLoader:resetIterator(split)
  assert(split == 0 or split == 1 or split == 2, 'split must be integer, either 0 (train), 1 (val) or 2 (test)')
  self.iterators[split] = 1
end

--[[
  split is an integer: 0 = train, 1 = val, 2 = test
  Returns a batch of data in two Tensors:
  - X (1,3,H,W) containing the image
  - B (1,R,4) containing the boxes for each of the R regions in xcycwh format
  - y (1,R,L) containing the (up to L) labels for each of the R regions of this image
  - info table of length R, containing additional information as dictionary (e.g. filename)
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
  Returning random examples is also supported by passing in .iterate = false in opt.
--]]
function DataLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split', 0)
  local iterate = utils.getopt(opt, 'iterate', true)

  assert(split == 0 or split == 1 or split == 2, 'split must be integer, either 0 (train), 1 (val) or 2 (test)')
  local split_ix
  if split == 0 then split_ix = self.train_ix end
  if split == 1 then split_ix = self.val_ix end
  if split == 2 then split_ix = self.test_ix end
  assert(#split_ix > 0, 'split is empty?')
  
  -- pick an index of the datapoint to load next
  local ri -- ri is iterator position in local coordinate system of split_ix for this split
  local max_index = #split_ix
  if self.debug_max_train_images > 0 then max_index = self.debug_max_train_images end
  if iterate then
    ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1 end -- wrap back around
    self.iterators[split] = ri_next
  else
    -- pick an index randomly
    ri = torch.random(max_index)
  end
  ix = split_ix[ri]
  assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)
  
  -- fetch the image
  local  img = self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                            {1,self.max_image_height},{1,self.max_image_width})

  -- crop image to its original width/height, get rid of padding, and dummy first dim
  img = img[{ 1, {}, {1,self.image_heights[ix]}, {1,self.image_widths[ix]} }]
  img = img:float() -- convert to float
  img = img:view(1, img:size(1), img:size(2), img:size(3)) -- batch the image
  img:csub(self.mean) -- subtract mean

  -- fetch the corresponding labels array
  local r0 = self.img_to_first_box[ix]
  local r1 = self.img_to_last_box[ix]
  local embeddings = self[self.embedding .. '_word_embeddings'][{ {r0,r1} }]
  local box_batch = self.boxes[{ {r0,r1} }]
  local labels = self.labels[{ {r0,r1} }]
  
  -- batch the boxes and labels and embeddings
  assert(box_batch:nDimension() == 2)
  embeddings = embeddings:view(1, embeddings:size(1), embeddings:size(2))
  box_batch = box_batch:view(1, box_batch:size(1), box_batch:size(2))

  -- finally pull the info from json file
  local w,h = self.image_widths[ix], self.image_heights[ix]
  local ow,oh = self.original_widths[ix], self.original_heights[ix]
  local info_table = { split_bounds = {ri, #split_ix}, width = w, height = h, ori_width = ow, ori_height = oh}

  r0 = self.img_to_first_rp[ix]
  r1 = self.img_to_last_rp[ix]
  local region_proposals = self.h5_file:read('/region_proposals'):partial({r0,r1}, {1, 4})
  region_proposals = region_proposals:view(1, region_proposals:size(1), region_proposals:size(2)) -- make batch

  return img, box_batch, region_proposals, embeddings, labels, info_table
end

