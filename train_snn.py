"""Train the VT-SNN models
"""
from pathlib import Path
import logging
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import slayerSNN as snn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()

parser = argparse.ArgumentParser("Train VT-SNN models.")

parser.add_argument(
    "--epochs", type=int, help="Number of epochs.", required=True
)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="Path for saving checkpoints.",
    default=".",
)

parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument(
    "--sample_file",
    type=int,
    help="Sample number to train from.",
    required=True,
)
parser.add_argument(
    "--finger_type", type=str, choices=["left", "right", "both"], help="finger type.", required=True
)

parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)
parser.add_argument("--variant", type=int, help="Variant Type.", required=True)
parser.add_argument(
    "--network_config",
    type=str,
    help="Path SNN network configuration.",
    required=True,
)

args = parser.parse_args()

# define dataset
class TacDataset(Dataset):
    def __init__(self, path, sample_file, output_size, finger_type='both', variant=1):
        self.path = path
        self.output_size = output_size
        self.samples = np.loadtxt(Path(path) / sample_file).astype("int")

        if variant == 1:
            tact = torch.load(Path(path) / "all.pt")
        elif variant == 2:
            tact = torch.load(Path(path) / "all_var2.pt")
            
        if finger_type == 'left':
            tact = tact[:,:39,...]
        elif finger_type == 'right':
            tact = tact[:,39:,...]
        
        self.tact = tact.reshape( tact.shape[0], -1, 1, 1, tact.shape[-1] )

    def __getitem__(self, index):
        input_index = self.samples[index, 0]
        class_label = self.samples[index, 1]
        target_class = torch.zeros((self.output_size, 1, 1, 1))
        target_class[class_label, ...] = 1

        return (
            self.tact[input_index],
            target_class,
            class_label,
        )

    def __len__(self):
        return self.samples.shape[0]


netParams = snn.params(args.network_config)
hidden_size = 16
device = torch.device('cuda:02')

input_size=78
if args.finger_type=='both':
    input_size *= 2


class SlayerMLP(torch.nn.Module):
    """2-layer MLP built using SLAYER's spiking layers."""

    def __init__(self, params, input_size, hidden_size, output_size):
        super(SlayerMLP, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc1 = self.slayer.dense(input_size, hidden_size)
        self.fc2 = self.slayer.dense(hidden_size, output_size)

    def forward(self, spike_input):
        spike_1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_output = self.slayer.spike(self.slayer.psp(self.fc2(spike_1)))
        return spike_output

net = SlayerMLP(netParams, input_size=input_size, hidden_size=hidden_size, output_size=4).to(device)
error = snn.loss(netParams).to(device)
writer = SummaryWriter(".")


optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=0.5)

val_dataset = TacDataset(
    path=args.data_dir,
    sample_file=f"val_{args.sample_file}.txt",
    output_size=4,
    finger_type=args.finger_type,
    variant=args.variant
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
)

train_dataset = TacDataset(
    path=args.data_dir,
    sample_file=f"train_{args.sample_file}.txt",
    output_size=4,
    finger_type=args.finger_type,
    variant=args.variant
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
)

test_dataset = TacDataset(
    path=args.data_dir,
    sample_file=f"test.txt",
    output_size=4,
    finger_type=args.finger_type,
    variant=args.variant
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
)


def _train():
    correct = 0
    num_samples = 0
    net.train()
    for data, target, label in train_loader:
        data = data.to(device)
        target = target.to(device)
        output = net.forward(data)
        correct += torch.sum(snn.predict.getClass(output) == label).data.item()
        num_samples += len(label)
        spike_loss = error.numSpikes(output, target)

        optimizer.zero_grad()
        spike_loss.backward()
        optimizer.step()
        
    writer.add_scalar("loss/train", spike_loss / len(train_loader), epoch)
    writer.add_scalar("acc/train", correct / num_samples, epoch)


def _val():
    correct = 0
    num_samples = 0
    net.eval()
    with torch.no_grad():
        for data, target, label in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = net.forward(data)
            correct += torch.sum(
                snn.predict.getClass(output) == label
            ).data.item()
            num_samples += len(label)
            spike_loss = error.numSpikes(output, target)  # numSpikes

        writer.add_scalar("loss/val", spike_loss / len(val_loader), epoch)
        writer.add_scalar("acc/val", correct / num_samples, epoch)

def _test():
    correct = 0
    num_samples = 0
    net.eval()
    with torch.no_grad():
        for data, target, label in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = net.forward(data)
            correct += torch.sum(
                snn.predict.getClass(output) == label
            ).data.item()
            num_samples += len(label)
            spike_loss = error.numSpikes(output, target)  # numSpikes

        writer.add_scalar("loss/test", spike_loss / len(test_loader), epoch)
        writer.add_scalar("acc/test", correct / num_samples, epoch)

def _save_model(epoch):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = Path(args.checkpoint_dir) / f"weights_{epoch:03d}.pt"
    model_path = Path(args.checkpoint_dir) / f"model_{epoch:03d}.pt"
    torch.save(net.state_dict(), checkpoint_path)
        
for epoch in range(1, args.epochs + 1):
    _train()
    if epoch % 10 == 0:
        _val()
        _test()
    if epoch % 100 == 0:
        _save_model(epoch)

with open("args.pkl", "wb") as f:
    pickle.dump(args, f)
