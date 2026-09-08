from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.perception.dataset import ReachPoseDataset
from src.perception.model import PoseRegressor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ReachPoseDataset("data/reach_pose", "train")
val_dataset = ReachPoseDataset("data/reach_pose", "val")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

model = PoseRegressor().to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

epochs = 10
best_val_loss = float("inf")

Path("models").mkdir(exist_ok=True)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(images)
            loss = criterion(predictions, targets)

            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_dataset)

    print(
        f"epoch {epoch + 1}/{epochs} "
        f"train_loss={train_loss:.6f} "
        f"val_loss={val_loss:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "models/reach_pose_resnet18_tightcam_10k.pt",
        )

print("saved: models/reach_pose_resnet18_tightcam_10k.pt")