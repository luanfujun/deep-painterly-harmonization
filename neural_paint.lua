require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'
require 'libcuda_utils'

require 'cutorch'
require 'cunn'

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg', 'Style target image')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg', 'Content target image')
cmd:option('-cnnmrf_image', 'examples/inputs/cnnmrf.jpg', 'CNNMRF image')
cmd:option('-tmask_image', 'examples/inputs/t_mask.jpg', 'Content tight mask image')
cmd:option('-mask_image', 'examples/inputs/t_mask.jpg', 'Content loose mask image')
cmd:option('-image_size', 700, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization optins
cmd:option('-init', 'image', 'random|image')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 0.1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-index', 0)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-original_colors', 0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', 316)
cmd:option('-num_iterations', 1000)

-- Patchmatch
cmd:option('-patchmatch_size', 3)
-- RefineNNF 
cmd:option('-refine_size', 5)
cmd:option('-refine_iter', 1)
-- Ring 
cmd:option('-ring_radius', 1) 
-- Wiki Art
cmd:option('-wikiart_fn', 'wikiart_output.txt')

local function main(params)
  cutorch.setDevice(params.gpu + 1)
  cutorch.setHeapTracking(true)

  torch.manualSeed(params.seed)

  idx = cutorch.getDevice()
  print('Gpu, idx      = ', params.gpu, idx)
  
  local layers         = string.format('relu1_1,relu2_1,relu3_1,relu4_1'):split(",")
  local content_layers = string.format('relu4_1'):split(",")  
  local style_layers   = string.format('relu1_1,relu2_1,relu3_1,relu4_1'):split(",")
  local hist_layers    = string.format('relu1_1,relu4_1'):split(",") 
  local content_weight = 1.0 
  local style_weight   = 1.0 
  local hist_weight    = 1.0 
  local tv_weight      = 1.0  
  local num_iterations = params.num_iterations

  local content_image       = image.load(params.content_image, 3)
  content_image             = image.scale(content_image, params.image_size, 'bilinear') 
  local content_image_caffe = preprocess(content_image):float():cuda()
  
  local style_image         = image.load(params.style_image, 3)
  style_image               = image.scale(style_image, params.image_size, 'bilinear')
  local style_image_caffe   = preprocess(style_image):float():cuda()
  
  local cnnmrf_image       = image.load(params.cnnmrf_image, 3)
  cnnmrf_image             = image.scale(cnnmrf_image, params.image_size, 'bilinear')
  local cnnmrf_image_caffe = preprocess(cnnmrf_image):float():cuda()

  -- Loose mask 
  local mask_image     = image.load(params.mask_image, 3)[1]
  mask_image           = image.scale(mask_image, params.image_size, 'bilinear'):float()
  local mask_image_ori = mask_image:clone()

  -- Tight mask
  local tmask_image         = image.load(params.tmask_image, 3)
  tmask_image               = image.scale(tmask_image, params.image_size, 'bilinear'):float()
  local tmask_image_ori     = tmask_image:clone()

  local tr                  = 3;
  local tkernel             = image.gaussian(2*tr+1, tr, 1, true):float()
  tmask_image               = image.convolve(tmask_image, tkernel, 'same')
  
  -- Note: Modify here for custom painting composites  
  --       or use our pre-trained model (coming soon) on wikiart dataset... 
  style_weight, hist_weight, tv_weight = params_wikiart_genre(style_image, params.index, params.wikiart_fn)
  -- content_weight = 1.0
  -- style_weight   = 1.0
  -- hist_weight    = 1.0 
  -- tv_weight      = 10.0
  
  -- load VGG-19 network
  local cnn = loadcaffe.load(params.proto_file, params.model_file, params.backend):float():cuda()

  local feature_extractor = nn.Sequential()
  local input_features, target_features, match_features, match_masks = {}, {}, {}, {}
  local layerIdx = 1
  for i = 1, cnn:size() do
    if layerIdx <= #layers then
      local layer = cnn:get(i)
      local name = layer.name
      feature_extractor:add(layer)
      if name == layers[layerIdx] then
        print("Extracting feature layer ", i, ":", layer.name)
        local input  = feature_extractor:forward(cnnmrf_image_caffe):clone()
        local target = feature_extractor:forward(style_image_caffe):clone()
        table.insert(input_features, input)
        table.insert(target_features, target)
        layerIdx = layerIdx + 1
      end 
    end
  end
  feature_extractor = nil
  collectgarbage()


  -- Feature matching & manipulation 
  local curr_corr, corr = nil, nil
  local curr_mask, mask = nil, nil 

  for i = #layers, 1, -1 do 
    local name = layers[i]
    print("Working on patchmatch layer ", i, ":", name)
    local A    = input_features[i]:clone()
    local BP   = target_features[i]:clone() 
    local N_A  = normalize_features(A)
    local N_BP = normalize_features(BP)

    local c, h, w = A:size(1), A:size(2), A:size(3)
    local _, h2, w2 = BP:size(1), BP:size(2), BP:size(3)
    if h ~= h2 or w ~= w2 then 
      print("  Input and target should have the same dimension! h, h2, w, w2 = ", h, h2, w, w2)
    end 

    local tmask = image.scale(torch.gt(tmask_image_ori[1], 0.1), w, h, 'simple'):cudaInt()
    if i == #layers then -- i = 5, relu5_1
      print("  Initializing NNF in layer ", i, ":", name, " with patch ", params.patchmatch_size) 
      print("  Brute-force patch matching...")
      local init_corr = cuda_utils.patchmatch(N_A, N_BP, params.patchmatch_size)
      local guide = image.scale(style_image, w, h, 'bilinear'):float():cuda()
      print("  Refining NNF...")
      corr = cuda_utils.refineNNF(N_A, N_BP, init_corr, guide, tmask, params.refine_size, params.refine_iter)
      mask = cuda_utils.Ring2(N_A, N_BP, corr, params.ring_radius, tmask)
      
      curr_corr = corr 
      curr_mask = mask       
    else -- i = 4, relu4_1
      print("  Upsampling NNF in layer ", i, ":", name)
      curr_corr = cuda_utils.upsample_corr(corr, h, w)
      curr_mask = image.scale(mask:double(), w, h, 'simple'):cudaInt()
    end   
      
    table.insert(match_features, BP)
    table.insert(match_masks, curr_mask)
  end 

  local gram_features, hist_features    = {}, {}
  local gram_match_masks, hist_match_masks = {}, {}
  local gramIdx, histIdx = 1, 1
  for i = 1, #layers do 
    local name = layers[i]
    local features = match_features[#layers - i + 1]
    local mask     = match_masks[#layers - i + 1]
    if gramIdx <= #style_layers or histIdx <= #hist_layers then 
      if name == style_layers[gramIdx] then 
        table.insert(gram_features, features)
        table.insert(gram_match_masks, mask)
        gramIdx = gramIdx + 1
      end 
      if name == hist_layers[histIdx] then 
        table.insert(hist_features, features)
        table.insert(hist_match_masks, mask)
        histIdx = histIdx + 1
      end 
    end 
  end 
  input_features = nil 
  target_features = nil 
  collectgarbage()


  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses, hist_losses = {}, {}, {}
  local next_content_idx, next_style_idx, next_hist_idx = 1, 1, 1
  local net = nn.Sequential()


  local tv_mod = nn.TVLoss(tv_weight, mask_image):float():cuda()
  net:add(tv_mod)

  for i = 1, cnn:size() do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers or next_hist_idx <= #hist_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      local is_conv    = (layer_type == 'nn.SpatialConvolution' or layer_type == 'cudnn.SpatialConvolution')
      
      net:add(layer)

      if is_pooling then
        mask_image = image.scale(mask_image, math.ceil(mask_image:size(2)/2), math.ceil(mask_image:size(1)/2))
      elseif is_conv then
        local sap = nn.SpatialAveragePooling(3,3,1,1,1,1):float()
        mask_image = sap:forward(mask_image:repeatTensor(1,1,1))[1]:clone()
      end

      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local input = net:forward(content_image_caffe):clone()
        local mask  = mask_image:float():repeatTensor(1,1,1):expandAs(input):cuda()
        local loss_module = nn.ContentLoss(content_weight, input, mask):float():cuda()
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end

      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram   = GramMatrix():float():cuda()
        local input  = net:forward(cnnmrf_image_caffe):clone()
        local target = net:forward(style_image_caffe):clone()
        local mask   = mask_image:clone():repeatTensor(1,1,1):expandAs(target):cuda()

        local c, h1, w1 = input:size(1), input:size(2), input:size(3)
        local _, h2, w2 = target:size(1), target:size(2), target:size(3)


        local gram_feature = gram_features[next_style_idx]
        local gram_mask    = gram_match_masks[next_style_idx] --mask_image --torch.ones(h1, w1) --gram_match_masks[next_style_idx]
        local gram_msk     = gram_mask:float():repeatTensor(1,1,1):expandAs(input):cuda()
        local target_gram  = gram:forward(torch.cmul(gram_feature, gram_msk)):clone()
        target_gram:div(gram_mask:sum() * c)

        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(style_weight, target_gram, mask):float():cuda()
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1

        if name == hist_layers[next_hist_idx] then 
          print("Setting up histogram layer", i, ":", layer.name)
          local maskI = torch.gt(mask_image, 0.1)
          local maskJ = hist_match_masks[next_hist_idx]:byte() -- maskI:clone() --torch.ones(h1, w1):byte() -- 
          local hist_feature = hist_features[next_hist_idx]
          local loss_module = nn.HistLoss(hist_weight, input, hist_feature, 256, maskI, maskJ, mask_image):float():cuda()
          net:add(loss_module)
          table.insert(hist_losses, loss_module)
          next_hist_idx = next_hist_idx + 1
        end 

      end
    end
  end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
  
  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():cuda():mul(0.001)
  elseif params.init == 'image' then
    img = content_image_caffe:clone():float():cuda()
  else
    error('Invalid init type')
  end
   
  
  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = num_iterations,
      verbose=true,
      tolX=-1,
      tolFun=-1,
      learningRate=params.learning_rate,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, num_iterations))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content   %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style     %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(hist_losses) do 
        print(string.format('  Histogram %d loss: %f', i, loss_module.loss))
      end 
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == num_iterations
    if should_save then
      -- local disp = deprocess(img:double())
      local disp = torch.cmul(img:double(), tmask_image:double())
      disp:add(torch.cmul(style_image_caffe:double(), 1.0 - tmask_image:double()))
      disp = deprocess(disp)
      disp = image.minmax{tensor=disp, min=0, max=1}

      local filename = build_filename(params.output_image, t)
      if t == num_iterations then
        filename = params.output_image
      end

      -- Maybe perform postprocessing for color-independent style transfer
      if params.original_colors == 1 then
        disp = original_colors(content_image, disp)
      end

      image.save(filename, disp)
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this fucntion many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:updateGradInput(x, dy)
    
    local msk = mask_image_ori:clone()
    msk = msk:repeatTensor(1,1,1):expandAs(x):cuda()
    grad:cmul(msk)

    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(hist_losses) do
      loss = loss + mod.loss
    end 
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, num_iterations do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  end
end

function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end

-- Combine the Y channel of the generated image and the UV channels of the
-- content image to perform color-independent style transfer.
function original_colors(content, generated)
  local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
  local content_uv = image.rgb2yuv(content)[{{2, 3}}]
  return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end

-- Normalize 3D feature map in channel dimension
function normalize_features(x)
  local c, h, w = x:size(1), x:size(2), x:size(3)
  print("Normalizing feature map with dim3[x] = ", c, h, w)
  local x2 = torch.pow(x, 2)
  local sum_x2 = torch.sum(x2, 1)
  local dis_x2 = torch.sqrt(sum_x2)
  local Nx = torch.cdiv(x, dis_x2:expandAs(x) + 1e-8)
  -- local Nx = torch.cdiv(x, dis_x2:expandAs(x))
  return Nx
end 

-- Compute weight map
function compute_weightMap(x)
  local c, h, w = x:size(1), x:size(2), x:size(3)
  print("Computing weight map with dim3[x] = ", c, h, w)
  local x2 = torch.pow(x, 2)
  local sum_x2 = torch.sum(x2, 1)[1]
  local sum_min, sum_max = sum_x2:min(), sum_x2:max()
  local wMap = (sum_x2 - sum_min) / (sum_max - sum_min + 1e-8)
  -- local wMap = (sum_x2 - sum_min) / (sum_max - sum_min)
  return wMap
end

-- Estimate noise level
function noise_estimate(input)
  local C, H, W = input:size(1), input:size(2), input:size(3)
  local x_diff = torch.zeros(3, H - 1, W - 1)
  local y_diff = torch.zeros(3, H - 1, W - 1)
  x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  local x_diff_sqr = torch.pow(x_diff, 2)
  local y_diff_sqr = torch.pow(y_diff, 2)
  local diff_sqr = (x_diff_sqr + y_diff_sqr) / 2
  local noise, idx = diff_sqr:view(diff_sqr:nElement()):median()
  return noise[1]
end 

function params_wikiart_genre(style_image, index, wikiart_fn)
  -- Estimate painting TV noise level 
  local tv_nosie = noise_estimate(style_image)
  local tv_weight      = 10.0 / (1.0 + torch.exp(1e4 * tv_nosie - 25.0))
  local hist_weight    = 1.0 
  local content_weight = 1.0
  local style_weight   = 1.0

  local fid     = io.open(wikiart_fn)
  local sty_idx = 0
  -- local label   = nil
  local sty_lev = nil
  for line in fid:lines() do 
    if sty_idx == index then 
      print(line)
      local terms = line:split("=")
      sty_lev = tonumber(terms[4])
    end 
    sty_idx = sty_idx + 1
  end 

  style_weight   = sty_lev
  hist_weight    = (10.0 - tv_weight) * sty_lev
  tv_weight      = tv_weight * sty_lev

  io.close(fid)

  return style_weight, hist_weight, tv_weight
end 

-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, input, msk)
  parent.__init(self)
  self.strength  = strength
  self.input     = torch.cmul(input, msk)
  self.loss      = 0
  self.crit      = nn.MSECriterion()
  self.msk       = msk
end

function ContentLoss:updateOutput(input)
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  self.loss = self.crit:forward(torch.cmul(input, self.msk), self.input) * self.strength
  self.gradInput = self.crit:backward(torch.cmul(input, self.msk), self.input)

  self.gradInput:cmul(self.msk)
  local magnitude = torch.norm(self.gradInput, 2)
  self.gradInput:div(magnitude + 1e-8)
  self.gradInput:mul(self.strength)

  self.gradInput:add(gradOutput)
  self.gradInput:cmul(self.msk)

  return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target_gram, msk)
  parent.__init(self)
  self.strength    = strength
  self.target_gram = target_gram
  self.loss        = 0
  self.gram        = GramMatrix()
  self.G           = nil
  self.crit        = nn.MSECriterion()
  self.msk         = msk
  self.msk_mean    = msk:mean()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(torch.cmul(input, self.msk))
  self.G:div(self.msk_mean * input:nElement())

  self.loss = self.crit:forward(self.G, self.target_gram)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target_gram)
  dG:div(self.msk_mean * input:nElement())

  self.gradInput = self.gram:backward(torch.cmul(input, self.msk), dG)
    
  self.gradInput:cmul(self.msk)
  local magnitude = torch.norm(self.gradInput, 2) 
  self.gradInput:div(magnitude + 1e-8)
  self.gradInput:mul(self.strength)

  self.gradInput:add(gradOutput)
  self.gradInput:cmul(self.msk)

  return self.gradInput
end

-- Histogram loss from: https://arxiv.org/pdf/1701.08893.pdf
local HistLoss, parent = torch.class('nn.HistLoss', 'nn.Module')

function HistLoss:__init(strength, input, target, nbins, maskI, maskJ, mask)
  parent.__init(self)
  self.strength        = strength
  self.loss            = 0
  self.nbins           = nbins
  self.maskI           = maskI 
  self.nI              = maskI:sum()

  local c, h1, w1      = input:size(1), input:size(2), input:size(3)
  self.msk             = self.maskI:float():repeatTensor(1,1,1):expandAs(input):cuda()
  self.msk_sub         = torch.cmul(torch.ones(c, h1, w1):float(), 1 - self.msk:float()):cuda()
  self.mask            = mask:float():repeatTensor(1,1,1):expandAs(input):cuda()
  
  self.nJ              = maskJ:sum()
  local c, h2, w2      = target:size(1), target:size(2), target:size(3)
  local mJ             = maskJ:repeatTensor(1,1,1):expandAs(target)
  local J              = target:float()
  local _J             = J[mJ]:view(c, self.nJ)
  self.minJ, self.maxJ = _J:min(2), _J:max(2) 

  self.histJ           = cuda_utils.histogram(target, self.nbins, self.minJ:cuda(), self.maxJ:cuda(), maskJ:cuda()):float()

  self.histJ:mul(self.nI / self.nJ)
  self.cumJ            = torch.cumsum(self.histJ, 2) 

end 

function HistLoss:updateOutput(input)
  self.output = input
  return self.output
end

function HistLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()    
  
  local I = input
  local c, h1, w1 = I:size(1), I:size(2), I:size(3)
  local _I = torch.cmul(I, self.msk) - self.msk_sub
  local sortI, idxI = torch.sort(_I:view(c, h1*w1), 2)

  local R = I:clone()
  cuda_utils.hist_remap2(I, self.nI, self.maskI:cuda(), 
    self.histJ:cuda(), self.cumJ:cuda(), self.minJ, self.maxJ, 
    self.nbins, sortI:cuda(), idxI:cudaInt(), R)
  self.gradInput:add(I)
  self.gradInput:add(-1, R)   

  local err = self.gradInput:clone()
  err:pow(2.0)
  self.loss = err:sum() * self.strength / input:nElement()

  local magnitude = torch.norm(self.gradInput, 2)
  self.gradInput:div(magnitude + 1e-8)
  self.gradInput:mul(self.strength)

  self.gradInput:add(gradOutput)
  self.gradInput:cmul(self.mask)

  return self.gradInput
end 

local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength, mask)
  parent.__init(self)
  self.strength = strength
  self.mask     = mask 
  self.x_diff   = torch.Tensor()
  self.y_diff   = torch.Tensor()
  self.msk      = self.mask:clone():repeatTensor(3,1,1):cuda()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)

  self.gradInput:cmul(self.msk)
  local magnitude = torch.norm(self.gradInput, 2) 
  self.gradInput:div(magnitude + 1e-8)
  self.gradInput:mul(self.strength)

  self.gradInput:add(gradOutput)
  self.gradInput:cmul(self.msk)

  return self.gradInput
end

-- min
function min(x, y)
  local res = x 
  if res > y then 
    res = y 
  end 
  return res
end 

-- max 
function max(x, y)
  local res = x 
  if res < y then 
    res = y 
  end 
  return res 
end 

-- clamp 
function clamp(x, x_min, x_max)
  local res = x 
  if x < x_min then 
    res = x_min 
  end 
  if x > x_max then 
    res = x_max 
  end 
  return res
end 

local params = cmd:parse(arg)
main(params)
