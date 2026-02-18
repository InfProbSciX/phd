
import re, json, subprocess

loss_pattern = re.compile(r"step \d+: train loss [0-9\.]+, val loss ([0-9\.]+)")

best_losses = []
for i in range(1, 101):
    print(f"=== Run {i} ===")

    proc = subprocess.run(
        ['/home/ar847/.conda/envs/rxrx/bin/python', 'train.py', 'config/train_shakespeare_char.py'],
        capture_output=True,
        text=True,
    )

    losses = [float(m.group(1)) for m in loss_pattern.finditer(proc.stdout)]
    best_losses.append(min(losses))
    print(f"Run {i} best val loss: {best_losses[-1]}")

with open('results_summary.json', 'w') as f:
    json.dump(dict(best_losses=best_losses), f, indent=2)


### imagenet

import re, json, subprocess

acc_pattern = re.compile(
    r"Epoch\s+\d+:\s+Train\s+Acc:\s+([0-9\.]+)%,\s+Val\s+Acc:\s+([0-9\.]+)%"
)

best_val_accs = []
for i in range(10):
    print(f"=== Run {i} ===")

    proc = subprocess.run(
        ['/home/ar847/.conda/envs/rxrx/bin/python', 'imagenet.py'],
        capture_output=True,
        text=True,
    )

    accs = [float(m.group(2)) for m in acc_pattern.finditer(proc.stdout)]

    best_val = max(accs)
    best_val_accs.append(best_val)
    print(f"Run {i} best Val Acc: {best_val:.1f}%")

# write them out
with open('imagenet_eye.json', 'w') as f:
    json.dump(dict(best_val_accs=best_val_accs), f, indent=2)
