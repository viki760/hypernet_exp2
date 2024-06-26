import os
import gc
import torch
from torch.utils.data.dataloader import DataLoader, RandomSampler, Dataset
from tqdm import tqdm
from src.datamodule import CustomizedDataModule

DATASETS_PATH = "/data/_dataset/hash/dataset/"
PROC_DATASETS_PATH = "/data/_dataset/hash/dataset_proc"
datasets = ["cifar-10", "nus-wide-tc21", "flickr25k"]


def proc(d):
    print(f"Processing dataset {d}...")

    save_path = os.path.join(PROC_DATASETS_PATH, d)
    retrieval_img_save_path = os.path.join(save_path, "img_retrieval")
    os.makedirs(retrieval_img_save_path, exist_ok=True)

    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="conf"):
        args = compose(config_name="config")

    # print(OmegaConf.to_yaml(args))

    dm = CustomizedDataModule(d, DATASETS_PATH, args, None)
    dm.setup()

    query_dataset, retrieval_dataset, _ = dm.query_dataset, dm.retrieval_dataset, dm.train_dataset

    query_loader = DataLoader(
        query_dataset, 
        batch_size=args.batch_size, 
        num_workers=0,
        pin_memory=args.pin_memory,
        shuffle=False
    )

    retrieval_loader = DataLoader(
        retrieval_dataset, 
        batch_size=args.batch_size, 
        num_workers=0,
        pin_memory=args.pin_memory,
        shuffle=False
    )

    torch.save({
        "query_target": query_dataset.get_onehot_targets(),
        "retrieval_target": retrieval_dataset.get_onehot_targets()
    }, os.path.join(save_path, "target.pt"))

    query_data = []
    for batch in tqdm(query_loader):
        img = batch[0]
        query_data.append(img)
    query_data = torch.cat(query_data, dim=0)

    torch.save(query_data, os.path.join(save_path, "query_data.pt"))

    # # retrieval_data = [batch[0] for batch in tqdm(retrieval_loader)]
    # retrieval_data = []
    # for idx, batch in tqdm(enumerate(retrieval_loader)):
    #     if idx > 50:
    #         break
    #     img = batch[0].numpy()
    #     retrieval_data.append(img)
    #     del batch
    #     del img
    #     gc.collect()
    # retrieval_data = torch.cat(retrieval_data, dim=0)

    # torch.save({
    #     "query_data": query_data,
    #     "query_target": query_dataset.get_onehot_targets(),
    #     "retrieval_data": retrieval_data,
    #     "retrieval_target": retrieval_dataset.get_onehot_targets()
    # }, os.join(save_path, "data.pt"))


    for idx, (img, target, index) in tqdm(enumerate(retrieval_loader)):
        # for i, img in zip(index, img):
        #     torch.save(img, os.path.join(retrieval_img_save_path, f"{i}.pt"))
        torch.save(img, os.path.join(retrieval_img_save_path, f"batch_{idx}.pt"))

    print(f"Dataset {d} processed.")


    # query_data = [batch[0] for batch in tqdm(query_loader)]
    # query_data = []
    # from PIL import Image, ImageFile
    # for batch in tqdm(query_loader):

    #     # print(batch[0][0].shape)
    #     # cifar 10: 3*32*32
    #     # print(query_dataset.data[0].shape)
    #     # flicker & nuswide
    #     # img = Image.open(os.path.join(query_dataset.root, query_dataset.data[0])).convert('RGB')
    #     # print(img.size)
    #     return
    #     query_data.append(batch[0])

    
    # #占用内存太大了，32*32*3 int8 -> 224*224*3 float32
    # retrieval_data = [batch[0] for batch in tqdm(retrieval_loader)] 


if __name__ == "__main__":

    # dataset = datasets[0]
    # proc(dataset)
    # exit()
    # for d in datasets:
    #     proc(d)
    proc(datasets[0])
    proc(datasets[1])
    proc(datasets[2])
    print("All datasets processed.")

    # from torch.profiler import profile, record_function, ProfilerActivity
    # # with profile(activities=[ProfilerActivity.CPU],
    # with profile(activities=[ProfilerActivity.CPU],
    #     profile_memory=True, record_shapes=True, with_stack=True) as prof:
    #     proc(datasets[0])

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_memory_usage", row_limit=10))

    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

