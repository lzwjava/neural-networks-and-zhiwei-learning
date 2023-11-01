import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Sequence(nn.Module):

    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)

    def forward(self, input, future=0):
        outputs = []
        i0 = input.size(0)
        h_t = torch.zeros(i0, 51, dtype=torch.double)
        c_t = torch.zeros(i0, 51, dtype=torch.double)
        h_t2 = torch.zeros(i0, 51, dtype=torch.double)
        c_t2 = torch.zeros(i0, 51, dtype=torch.double)

        

        return []


def main():
    steps = 15
    np.random.seed(0)
    torch.manual_seed(0)
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    print(input)
    print(target)
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    seq = Sequence()
    seq.double()

    loss_fn = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    for i in range(steps):
        print('STEP: ', i)

        optimizer.zero_grad()
        out = seq(input)
        loss = loss_fn(out, target)
        print('loss: ', loss.item())
        loss.backward()

        optimizer.step()

        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = loss_fn(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()

        plt.figure(figsize=(30, 10))
        plt.title('Predict')
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            i1 = input.size(1)
            plt.plot(np.arange(i1), yi[:i1], color, linewidth=2.0)
            plt.plot(np.arange(i1, i1 + future), yi[i1:], color + ':', linewidth=2.0)

        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')

        plt.savefig('predict%d.pdf' % i)
        plt.close()


if __name__ == '__main__':
    main()
