from utils.distributed import launch_distributed_job
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from utils.dataset import TextDataset
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os


def init_model(device):
    model = WanDiffusionWrapper().to(device).to(torch.float32)
    encoder = WanTextEncoder().to(device).to(torch.float32)
    model.model.requires_grad_(False)

    base_checkpoint_path = "/hdd/u202420081000004/models/multi_shot_1000.pt"
    base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
                    
    # Load generator (directly; no key alignment needed since LoRA not applied yet)

       

    result = model.load_state_dict(base_checkpoint, strict=False)



    scheduler = FlowMatchScheduler(
        shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=48, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=6.0)

    args = parser.parse_args()

    # launch_distributed_job()
    launch_distributed_job()

    device = torch.cuda.current_device()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(device=device)

    dataset = TextDataset(args.caption_path)

    # if global_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

    if global_rank == 0:
        print("Scanning existing files to re-distribute workload...")

    all_indices = range(len(dataset))
    remaining_indices = []

    # 遍历所有任务，检查是否存在
    # 注意：这里我们只做一次扫描，启动时可能会稍微花几秒钟
    for i in all_indices:
        save_filename = f"{i:05d}.pt"
        save_path = os.path.join(args.output_folder, save_filename)
        
        # 如果文件不存在，或者文件大小为0（损坏），则加入待办列表
        if not (os.path.exists(save_path) and os.path.getsize(save_path) > 0):
            remaining_indices.append(i)

    if global_rank == 0:
        print(f"Total tasks: {len(dataset)}. Remaining tasks: {len(remaining_indices)}.")

        # ================== 【第二阶段：重新分配】 ==================
        # 将剩下的任务列表 remaining_indices 平均分配给当前的 GPU
        # 使用切片语法 list[start:end:step] 进行交错分配
        
    my_tasks = remaining_indices[global_rank::world_size]

    #for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
    for prompt_index in tqdm(my_tasks, disable=global_rank != 0, desc=f"Rank {global_rank}"):
        #prompt_index = index * dist.get_world_size() + dist.get_rank()
        prompt = dataset[prompt_index]
        if prompt_index >= len(dataset):
            continue
        data = dataset[prompt_index]
        prompt = data["prompts"]


        save_filename = f"{prompt_index:05d}.pt"
        save_path = os.path.join(args.output_folder, save_filename)

        # 3. 检查文件是否存在
        # 这里的 os.path.getsize(save_path) > 0 是为了防止上次崩溃留下的 0kb 空文件
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            # 如果存在，直接进入下一次循环，跳过推理
            print(f"Skipping prompt {prompt_index}")
            continue

        conditional_dict = encoder(text_prompts=prompt)

        latents = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device
        )

        noisy_input = []

        for progress_id, t in enumerate(tqdm(scheduler.timesteps)):
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)

            noisy_input.append(latents)

            _, x0_pred_cond = model(
                latents, conditional_dict, timestep
            )

            _, x0_pred_uncond = model(
                latents, unconditional_dict, timestep
            )

            x0_pred = x0_pred_uncond + args.guidance_scale * (
                x0_pred_cond - x0_pred_uncond
            )

            flow_pred = model._convert_x0_to_flow_pred(
                scheduler=scheduler,
                x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, x0_pred.shape[:2])

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones(
                    [1, 21], device=device, dtype=torch.long).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])



        noisy_input.append(latents)

        noisy_inputs = torch.stack(noisy_input, dim=1)

        noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -1]]

        stored_data = noisy_inputs

        torch.save(
            {prompt: stored_data.cpu().detach()},
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        )

    dist.barrier()


if __name__ == "__main__":
    main()
