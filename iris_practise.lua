require("torch")
require("nn")
require("nngraph")
require("optim")
RNN = {}

function RNN.create(opt)
    local x = nn.Identity()()
    local h_prev = nn.Identity()()
    local Wh = nn.Linear(opt.rnn_size,opt.rnn_size)({h_prev})
    local Ux = nn.Linear(opt.input_size,opt.rnn_size)({x})
    local a = nn.CAddTable()({Wh,Ux})
    local h = nn.Tanh()({a})

    local o = nn.Linear(opt.rnn_size,opt.input_size)({h})
    local y = nn.LogSoftMax()({o})
    return nn.gModule({x,h_prev},{h,y})
end
-- to make a model
function create_seq(opt)
    local model = nn.Sequential()
    model:add(nn.Linear(opt.size_in,opt.size_hidden))
    model:add(nn.Tanh())
    model:add(nn.Linear(opt.size_hidden,opt.size_out))
    model:add(nn.LogSoftMax())
    return model
end
--To define a function to fetch the dataset
local loader = {}
local lookup = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
local classname_to_index = {}
for k,v in pairs(lookup) do
    classname_to_index[v] = k
end
--change the name into index of the dataset 
function loader.classname_to_index(name)
    return classname_to_index[name]
end
-- loading the dataset 
function loader.load_data()
    -- load 
    local data = {}
    data.inputs = {}
    data.targets = {}
    data.targets_by_name = {}
    local f = torch.DiskFile('iris.data.csv','r')
    f:quiet()
    local line = f:readString("*l")
    while line ~= '' do
        f1,f2,f3,f4,classname = string.match(line,'([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)')
        data.inputs[#data.inputs + 1] = {tonumber(f1),tonumber(f2),tonumber(f3),tonumber(f4)}
        data.targets[#data.targets + 1] = loader.classname_to_index(classname)
        data.targets_by_name[#data.targets_by_name + 1] = classname
        line = f:readString("*l")
    end
    data.inputs = torch.Tensor(data.inputs)
    data.targets = torch.Tensor(data.targets)

    -- shuffle the dataset
    local shuffled_indices = torch.randperm(data.inputs:size(1)):long()
    data.inputs = data.inputs:index(1,shuffled_indices):squeeze()
    data.targets = data.targets:index(1,shuffled_indices):squeeze()
    print('------------------------------------------')
    print('Loaded DataSize:')
    print('inputs',data.inputs:size())
    print('targets',data.targets:size())
    print('------------------------------------------')
    return data
end
-- Here we use the linear model to train the dataset
function Train_With_Linear()
    -- the model parameters defination
    opt = {}
    opt.size_in = 4
    opt.size_out = 4
    opt.size_hidden = 20
    --create the model
    model = create_seq(opt)
    --criterion
    criterion = nn.ClassNLLCriterion()
    -- parameters
    params,gradParams = model:getParameters()
    -- dataset
    dataset = loader.load_data()
    x = dataset.inputs
    y = dataset.targets
    -- feval function
    function feval(params)
        gradParams:zero()
        local outputs = model:forward(x)
        local loss = criterion:forward(outputs,y)
        local dloss_doutputs = criterion:backward(outputs,y)
        model:backward(x,dloss_doutputs)
        return loss,gradParams
    end
    --Trainig Processing
    optimState = {
        learningRate = 0.01
    }
    for eopch = 1,1000 do
        optim.sgd(feval,params,optimState)
    end
    -- evaluate on dataset
    result = model:forward(x)
    val,idx = torch.max(result,2)
    mask = idx:eq(y:long())
    acc = mask:sum()/mask:size(1)
    print('accurate'..acc)
end
-- Here we use the RNN model to train the dataset
function Train_With_RNN()
    -- the model parameters defination
    opt = {}
    opt.input_size = 4
    opt.rnn_size = 5
    --create the model
    model = RNN.create(opt)
    --criterion
    criterion = nn.ClassNLLCriterion()
    -- parameters
    params,gradParams = model:getParameters()
    -- dataset
    dataset = loader.load_data()
    x = dataset.inputs
    h0 = torch.Tensor(150,opt.rnn_size):zero()
    input = {x,h0}
    y = dataset.targets
    -- feval function
    function feval(params)
        gradParams:zero()
        local output = model:forward(input)
        local h = output[1]
        local predict = output[2]
        local lost = criterion:forward(predict,y)
        local dloss_doutput = criterion:backward(predict,y)
        model:backward(input,{h0,dloss_doutput})
        return params,gradParams
    end
    -- Training processing
    optimState = {
        learningRate = 0.01
    }
    for eopch = 1,1000 do
        optim.sgd(feval,params,optimState)
    end
    result = model:forward(input)
    result = result[2]
    val,idx = torch.max(result,2)
    mask = idx:eq(y:long())
    acc = mask:sum()/mask:size(1)
    print('accurate'..acc)
    gradParams:zero()
    output = model:forward(input)
    h = output[1]
    predict = output[2]
    lost = criterion:forward(predict,y)
    dloss_doutput = criterion:backward(predict,y)
    meow = model:backward(input,{h0,dloss_doutput})
end
Train_With_Linear()
Train_With_RNN()
