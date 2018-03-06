require('torch')
require('nn')
require('gnuplot')
require('optim')

x_data = torch.linspace(-1,1,300)
y_data = 10 * torch.pow(x_data,3) - 6 * x_data + 8
y_data = y_data / y_data:max()
tip = 0.03
y_data = y_data - tip + tip * torch.randn(y_data:size())
gnuplot.plot({x_data,y_data,'+'})

x_data = x_data:reshape(x_data:size(1),1)
y_data = y_data:reshape(y_data:size(1),1)

in_size = 1
hidden_size = 300
out_size = 1
BpNet = nn.Sequential()
BpNet:add(nn.Linear(in_size,hidden_size))
BpNet:add(nn.ReLU6())
BpNet:add(nn.Linear(hidden_size,out_size))

print(BpNet)
criterion = nn.MSECriterion()
local params,gradParams = BpNet:getParameters()
local optimState = {learningRate = 0.03}
for epoch =1,15000 do
	local function fevel(params)
		gradParams:zero()
		local predict = BpNet:forward(x_data)
		local loss = criterion:forward(predict,y_data)
		print(loss)
		local loss_q = criterion:backward(predict,y_data)
		local gradInput = BpNet:backward(x_data,loss_q)
		x_t = x_data:reshape(x_data:size(1))
		y_t = predict:reshape(predict:size(1))
		z_t = y_data:reshape(y_data:size(1))
		gnuplot.plot({x_t,y_t,'-'},{x_t,z_t,'+'})
		return loss,gradParams
	end
	optim.sgd(fevel,params,optimState)
end
