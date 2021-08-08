import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from datasets.speech_commands import SpeechCommands
from models.keyword_recognition import Keyword

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':

    batch_size = 100
    learning_rate = 0.001

    train_set = SpeechCommands('training')
    test_set = SpeechCommands('testing')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    labels = sorted(list(set(item[2] for item in train_set)))


    def collate_fn(batch):
        tensors, targets = [], []
        for waveform, _, label, *_, in batch:
            tensors += [waveform]
            targets += [label_to_index(label)]

        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets


    def label_to_index(word):
        return torch.tensor(labels.index(word))


    def index_to_label(index):
        return labels[index]


    def pad_sequence(batch):
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        return batch.permute(0, 2, 1)


    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # instantiate network (which has been imported from *networks.py*)
    model = Keyword(n_categories=len(labels))

    # create losses (criterion in pytorch)
    criterion_L1 = torch.nn.L1Loss()
    criterion = nn.CrossEntropyLoss()

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # create optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


    def train(model, epoch, log_interval):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            pbar.update(pbar_update)
            losses.append(loss.item())


    def number_of_correct(pred, target):
        return pred.squeeze().eq(target).sum().item()


    def get_likely_index(tensor):
        return tensor.argmax(dim=-1)


    def test(model, epoch):
        model.eval()
        correct = 0
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)

            pbar.update(pbar_update)

        print(
            f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


    log_interval = 20
    epochs = 20

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []

    with tqdm(total=epochs) as pbar:
        for epoch in range(1, epochs + 1):
            train(model, epoch, log_interval)
            test(model, epoch)
            scheduler.step()
        torch.save(model.state_dict(), 'model.pth')
