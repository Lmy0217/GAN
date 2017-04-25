require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'optim'


gray = torch.load('data/gray.t7')
speckle = torch.load('data/speckle.t7')

accsInterval = 0.9
datasize_G = 3884
datasize_D = 3884*2
batchsize = 1
split =  torch.FloatTensor(batchsize, 1, 256, 256):cuda()
sl = torch.FloatTensor(batchsize):cuda()

function batch_G(i)
    local t = i or 0
    local size = math.min(t+batchsize, datasize_G)-t
    if(size~=split:size(1)) then
       split = torch.FloatTensor(size, 1, 256, 256):cuda();
       sl = torch.FloatTensor(size):cuda();
    end
    for k=1,size do
       split[k][1] = speckle[t+k]:cuda()
       sl[k] = 1
    end

    return {split, sl}, t+size
end


function batch_D(i)
    local t = i or 0
    local size = math.min(t+batchsize, datasize_D)-t
    if(size~=split:size(1)) then
       split = torch.FloatTensor(size, 1, 256, 256):cuda();
       sl = torch.FloatTensor(size):cuda();
    end
    for k=1,size do
       if((t+k)%2==0) then
          split[k][1] = gray[(t+k)/2]:cuda()
          sl[k] = 1
       else
          split[k][1] = speckle[(t+k+1)/2]:cuda()
          sl[k] = -1
       end
    end

    return {split, sl}, t+size
end


MODEL_D = torch.load('D0.t7'):cuda()
cudnn.convert(MODEL_D, cudnn)
MODEL_G = torch.load('G0.t7'):cuda()
cudnn.convert(MODEL_G, cudnn)

CRITERION = nn.BCECriterion():cuda()
PARAMETERS_D, GRAD_PARAMETERS_D = MODEL_D:getParameters()
PARAMETERS_G, GRAD_PARAMETERS_G = MODEL_G:getParameters()

trainDflag = true
local fevalD = function(x)
    if x ~= PARAMETERS_D then
       --PARAMETERS_D:copy(x)
    end

    GRAD_PARAMETERS_D:zero()

    local outputs = MODEL_D:forward(batch[1])
    
    for n=1,outputs:size(1) do
        if((outputs[n][1] >= 0.5 and batch[2][n] == 1) or (outputs[n][1] < 0.5 and batch[2][n] == -1)) then
            account = account + 1
        end
    end
    
    local f = CRITERION:forward(outputs, batch[2])
    erD = erD + f;
    local df_do = CRITERION:backward(outputs, batch[2])
    MODEL_D:backward(batch[1], df_do)

    return f,GRAD_PARAMETERS_D
end

PARAMETERS = torch.CudaTensor(PARAMETERS_D:size())
local trainD = function()
    account = 0
    indexD = 0
    erD = 0
    local maxIterate = torch.ceil(datasize_D/batchsize)
    for i=1,maxIterate do
        batch, indexD = batch_D(indexD)
        if(trainDflag) then
           optim.sgd(fevalD, PARAMETERS_D, optimState)
        else
           optim.sgd(fevalD, PARAMETERS, optimState)
        end
    end
    erD = erD/maxIterate
    if((account/datasize_D) > accsInterval) then
        trainDflag = false
    else
        trainDflag = true
    end
    print('..accuracy ' .. (account/datasize_D))
    print('.... ' .. erD)
    collectgarbage();
end

local fevalG = function(x)
    if x ~= PARAMETERS_G then
       PARAMETERS_G:copy(x)
    end

    GRAD_PARAMETERS_G:zero()

    local samples = MODEL_G:forward(batch[1])
    local outputs = MODEL_D:forward(samples)
    local f = CRITERION:forward(outputs, batch[2])
    erG = erG + f
    local df_samples = CRITERION:backward(outputs, batch[2])
    MODEL_D:backward(samples, df_samples)
    local df_do = MODEL_D.modules[1].gradInput
    MODEL_G:backward(batch[1], df_do)

    return f,GRAD_PARAMETERS_G
end

local trainG = function()
    indexG = 0
    erG = 0
    local maxIterate = torch.ceil(datasize_G/batchsize)
    for i=1,maxIterate do
        batch, indexG = batch_G(indexG)
        optim.sgd(fevalG, PARAMETERS_G, optimState)
    end
    erG = erG/maxIterate
    print('.... ' .. erG)
    collectgarbage();
end

io.output("log.txt")
torch.setdefaulttensortype('torch.FloatTensor')

local optimState = {learningRate = 0.001}
for k=1,100 do
    print('epoch: ' .. k)

    batchsize = 4
    print('.. train D')
    trainD()
    MODEL_D = MODEL_D:clearState()
    torch.save('models/D' .. (k) .. '.t7', MODEL_D:clearState())

    batchsize = 1
    print('.. train G')
    trainG()
    MODEL_G = MODEL_G:clearState()
    torch.save('models/G' .. (k) .. '.t7', MODEL_G:clearState())
end
