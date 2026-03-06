# Complete Guide: Converting a Bootable USB Pendrive to Normal Storage

---

## Overview

This guide provides step-by-step instructions to convert a **bootable USB pendrive** into a **normal storage device** by removing the boot partition, EFI system partition, and boot flags, then creating a single FAT32 partition.

---

## Prerequisites

- Ubuntu/Linux terminal access
- A bootable USB pendrive (64GB in this example)
- `sudo` privileges
- Basic knowledge of terminal commands

---

## ⚠️ Important Warnings

- **Back up any important data** on the pendrive before proceeding
- **Identify the correct device name** (e.g., `/dev/sdb`) to avoid data loss on other drives
- All data on the pendrive **will be permanently deleted**
- Do not interrupt the process once started

---

## Standard Procedure

### Step 1: Identify Your Pendrive

#### Command:
```bash
lsblk
```

or

```bash
sudo fdisk -l
```

#### What to Look For:
- Device name (e.g., `/dev/sdb`, `/dev/sdc`)
- Size (should match your pendrive capacity)
- Current partitions and their types

#### Example Output:
```
sdb                                                                         
├─sdb1        iso966 Jolie Ubuntu 25.04 amd64 2025-04-15-18-45-54-00
├─sdb2        vfat   FAT12 ESP                8F89-1BFD
└─sdb3        (empty partition)
```

---

### Step 2: Wipe the GPT Partition Table

#### Why This Step?
The bootable pendrive uses **GPT (GUID Partition Table)** with EFI boot information. We need to erase this completely before creating a new MBR partition table.

#### Command:
```bash
sudo dd if=/dev/zero of=/dev/sdb bs=1M count=10
```

#### What This Does:
- Writes zeros to the first 10MB of the device
- Destroys the GPT partition table and boot signatures
- Takes only a few seconds

#### Expected Output:
```
760 bytes (10 MB, 10 MiB) copied, 0.0110463 s, 953 MB/s
```

---

### Step 3: Unmount All Partitions

#### Why This Step?
If any partitions are mounted, the system won't allow you to modify them. You must unmount them first.

#### Commands:
```bash
sudo umount /media/anand/Ubuntu\ 25.04\ amd64
sudo umount /dev/sdb1
sudo umount /dev/sdb2
sudo umount /dev/sdb3
```

#### Expected Output:
```
umount: /dev/sdb1: not mounted.
umount: /dev/sdb2: not mounted.
umount: /dev/sdb3: not mounted.
```

> **Note:** "Not mounted" messages are fine — it means the partitions are already unmounted.

---

### Step 4: Create a New MBR Partition Table

#### Why This Step?
We replace the GPT table with a simpler **MBR (Master Boot Record)** table, which is standard for normal USB drives.

#### Command:
```bash
sudo fdisk /dev/sdb
```

#### Inside `fdisk`, Follow These Steps:

| Step | Action | Key Press |
|------|--------|-----------|
| 1 | Create new DOS (MBR) partition table | Press `o` |
| 2 | Create new partition | Press `n` |
| 3 | Select primary partition | Press `p` |
| 4 | Set partition number | Press `1` |
| 5 | Accept default first sector | Press **Enter** |
| 6 | Accept default last sector (full disk) | Press **Enter** |
| 7 | Change partition type to FAT32 | Press `t` → Type `b` |
| 8 | Remove bootable flag | Press `a` → Enter `1` |
| 9 | Write changes and exit | Press `w` |

#### Expected Output:
```
Welcome to fdisk (util-linux 2.41).
Changes will remain in memory only, until you decide to write them.

Command (m for help): o
Created a new DOS (MBR) disklabel with disk identifier 0x3c711d5c.

Command (m for help): n
Partition type
   p   primary (0 primary, 0 extended, 4 free)
   e   extended (container for logical partitions)
Select (default p): p
Partition number (1-4, default 1): 1
First sector (2048-122879999, default 2048): 
Last sector, +/-sectors or +/-size{K,M,G,T,P} (2048-122879999, default 122879999): 

Created a new partition 1 of type '
