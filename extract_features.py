# extract_features.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import os
import argparse

# --- 自定义 Dataset 以安全获取路径 ---
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        # 返回 (image, label, path)
        return original_tuple + (path,)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main(args):
    # Setup PyTorch Distributed
    assert torch.cuda.is_available(), "Extraction requires CUDA."
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
    else:
        rank = 0
        world_size = 1
        gpu = 0
        torch.cuda.set_device(gpu)

    device = torch.device(f"cuda:{gpu}")

    # Load VAE model
    if rank == 0:
        print(f"Loading VAE model from {args.vae_path}...")
    
    # 修复：直接使用 args.vae_path，避免字符串拼接错误
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # 移除 RandomHorizontalFlip，保证离线特征确定性
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # 使用自定义 Dataset
    dataset = ImageFolderWithPaths(args.data_path, transform=transform)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False, # 提取特征通常不需要 shuffle，顺序读取更利于 debug，且不影响结果
        seed=args.global_seed
    )
    
    # 修复：batch_size 参数逻辑
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size, # 这里假设传入的是 per-gpu batch size
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False # 提取特征不要 drop_last，否则会丢弃最后几张图
    )

    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        print(f"Total dataset size: {len(dataset)}")
        print(f"Features will be saved to: {args.features_path}")

    if world_size > 1:
        dist.barrier()

    print(f"Rank {rank}: Starting extraction...")
    
    with torch.no_grad():
        # loader 现在返回 (x, y, paths)
        for i, (x, y, paths) in enumerate(loader):
            x = x.to(device)
            
            # VAE Encode
            # Map input images to latent space + normalize latents
            # 注意：sample() 带有随机性。如果需要确定性特征，可以使用 .mode()
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            x = x.detach().cpu().numpy() # [B, 4, H/8, W/8]
            
            # 保存逻辑
            for b in range(x.shape[0]):
                path = paths[b] # 直接从 loader 获取真实路径，绝对安全
                
                # 构建输出路径
                rel_path = os.path.relpath(path, args.data_path)
                rel_path_no_ext = os.path.splitext(rel_path)[0]
                save_path = os.path.join(args.features_path, rel_path_no_ext + ".npy")
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, x[b])
                
            if i % 100 == 0:
                print(f"Rank {rank}: Processed batch {i}/{len(loader)}")

    print(f"Rank {rank}: Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, required=True)
    # 修复：默认值设为完整路径
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
    
# torchrun --nproc_per_node=4 extract_features.py --data-path /work/c30778/dataset/imagenet --features-path /work/c30778/dataset/dit_feat_fix