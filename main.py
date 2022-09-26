from Net import *
from data_prepare import *
from torch.utils.data import random_split
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings('ignore')

if __name__ == '__main__':
    epochs = 10
    batch_size = 2
    model = All_Net()
    if torch.cuda.is_available():
        print(1)
        model = model.cuda()
    EGG_MAT_list, PYS_MAT_list, LABEL_list = Data_path()
    loss_fn = nn.CrossEntropyLoss()  # 选择MSE作为损失loss
    if torch.cuda.is_available():	
        print(2)
        loss_fn = loss_fn.cuda()
    learning_rate = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    dataset = MyDataset(EGG_MAT_list, PYS_MAT_list, LABEL_list)
    train_size = int(len(dataset) * 0.8)  # 这里train_size是一个长度矢量，并非是比例，我们将训练和测试进行8/2划分
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    for it in range(epochs):
        train_loss = 0
        for step, (batch_x1,batch_x2,batch_x3, batch_y) in enumerate(tqdm((train_dataloader))):
            if torch.cuda.is_available():	
                batch_x1 = batch_x1.cuda()
                batch_x2 = batch_x2.cuda()
                batch_x3 = batch_x3.cuda()
                batch_y = batch_y.cuda()
            # Forward pass，前向网络
            y_pred = model(batch_x1,batch_x2,batch_x3)
            # model.forward()
            # compute loss，损失函数
            print(y_pred)
            loss = loss_fn(y_pred, batch_y)  # computation graph

            optimizer.zero_grad()  # 梯度归零
            # Backward pass，求导
            train_loss += loss.item()
            loss.backward()

            # update model parameters，参数更新
            optimizer.step()

        print('Epoch {} : ,Train_loss : {}'.format(it,train_loss))

    with torch.no_grad():  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
        # 测试
        total_loss = 0.
        acc_count = 0
        for step, (batch_x1,batch_x2,batch_x3, batch_y) in enumerate(test_dataloader):
            # Forward pass，前向网络
            y_pred = model(batch_x1,batch_x2,batch_x3)
            acc_count+= (np.argmax(y_pred,axis=1) == batch_y).sum().item()
            total_loss += loss_fn(y_pred, batch_y)
            print('Test accuracy is : {}%'.format(acc_count*100/len(test_dataloader)))