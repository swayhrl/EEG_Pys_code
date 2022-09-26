from Net import *
from data_prepare import *
from torch.utils.data import random_split
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings('ignore')

if __name__ == '__main__':
    epochs = 10
    batch_size = 4
    model = All_Net()
    #if torch.cuda.is_available():
    #   print(cuda)
    #   model = model.cuda()
    EGG_MAT_list, PYS_MAT_list, LABEL_list = Data_path()
    loss_fn = nn.CrossEntropyLoss()  # 选择MSE作为损失loss
    #if torch.cuda.is_available():	
    #    loss_fn = loss_fn.cuda()
    learning_rate = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    dataset = MyDataset(EGG_MAT_list, PYS_MAT_list, LABEL_list)
    train_size = int(len(dataset) * 0.7)  #我们将训练和测试进行7/3划分
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    for it in range(epochs):
        train_loss = 0
        for step, (batch_x1,batch_x2,batch_x3, batch_y) in enumerate(tqdm((train_dataloader))):
            #if torch.cuda.is_available():	
            #    batch_x1 = batch_x1.cuda()
            #    batch_x2 = batch_x2.cuda()
            #    batch_x3 = batch_x3.cuda()
            #    batch_y = batch_y.cuda()
            # Forward pass，前向网络
            y_pred = model(batch_x1,batch_x2,batch_x3)
            # model.forward()
            # compute loss，损失函数
            loss = loss_fn(y_pred, batch_y)  # computation graph

            optimizer.zero_grad()  # 梯度归零
            # Backward pass，求导
            train_loss += loss.item()
            loss.backward()

            # update model parameters，参数更新
            optimizer.step()

        print('Epoch {} : ,Train_loss : {}'.format(it,train_loss))

        torch.save(model, 'model.pkl')
        if((it % 2)== 0):
            with torch.no_grad():  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
                # 测试
                total_loss = 0
                acc_count = 0
                print("len(test_dataloader):",len(test_dataloader))
                for step, (batch_x1,batch_x2,batch_x3, batch_y) in enumerate(test_dataloader):
                    # Forward pass，前向网络
                    y_pred = model(batch_x1,batch_x2,batch_x3)
                    print(np.argmax(y_pred,axis=1))
                    print("batch_y:",batch_y)
                    acc_count+= (np.argmax(y_pred,axis=1) == batch_y).sum().item()
                    #print("acc_count:",acc_count)
                    #total_loss += loss_fn(y_pred, batch_y)
                print('Test accuracy is : {}%'.format(acc_count*100/(len(test_dataloader)*batch_size)))
