import torch
import torch.distributed as dist
from model import ChineseTokenizer,PrefixLM
import argparse
from torch.utils.data import DataLoader
from loader import TextImageDataset
import numpy as np
import random
import os
parser = argparse.ArgumentParser()
parser.add_argument("--d_model", type=int, default=512,help='embedding dimmension for transformer')
parser.add_argument('--heads', type=int, default=8,help='how many numbers of heads in MultiheadedAttention')
parser.add_argument('--enc_depth', type=int, default=8,help='depth of encoder')
parser.add_argument('--dec_depth', type=int, default=8,help='depth of decoder')
parser.add_argument('--d_ff', type == int, default=1024,help='hidden dimension of feedforward net')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=0,help='num of wokers for dataloader')
parser.add_argument('--num_epochs', type=int, default=50,help='how many epochs to train')
parser.add_argument('--input_resolution', type=int, default=224,help='image resolution')
parser.add_argument('--patch_size', type=int, default=16,help='size of one patch for the images')
parser.add_argument('--txt_seq_len', type=int, default=256,help='max len of texts')
parser.add_argument('--res_depth', type=int, default=3,help='depth of resnet')
parser.add_argument('--truncate_captions',type=bool,default=True,help='whether to truncate the captions when they are too long')
parser.add_argument('--shuffle',type=bool,default=False,help='whether permute the order of samples')
parser.add_argument('--data_folder', type=str, required=True,help='path to your data folder')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',help='path to your save checkpoint')
parser.add_argument('--prefixLM_path',type=str,help='path to your partially trained PrefixLM')
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--save_every_n_steps', default=1000, type=int, help='Save a checkpoint every n steps')
args = parser.parse_args()

def exists(val):
    return val is not None
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def configure_optimizers(model):
        lr =5e-4
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(
                0.9,
                0.999
            ),
            eps=1e-6,
            weight_decay=0.1
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2000
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def train_one_epoch(epoch,model,dataloader,data_sampler,optimizer,lr_scheduler,save_every_n_steps,rank):
    torch.cuda.empty_cache()
    data_sampler.set_epoch(epoch)
    for i, (txt, img) in enumerate(dataloader):
        txt=txt.cuda()
        img=img.cuda()
        loss=model(img,txt,return_loss=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%10==0 and rank==0:
            print('loss----------',loss)
        if i%save_every_n_steps==0 and rank==0:
            save_model(epoch,args.checkpoint_dir)
    if rank==0: save_model(epoch, args.checkpoint_dir)
    lr_scheduler.step()

#############################进程组#################
rank = int(os.environ['RANK'])  #获取当前进程号
world_size=int(os.environ['WORLD_SIZE'])
dist.init_process_group(
   backend='nccl',
   init_method='env://',
   world_size=world_size,
   rank=rank
   )#初始化
assert dist.is_initialized()
print('进程组初始化完成')
set_seed(0)
torch.cuda.set_device(args.local_rank)

start_epoch=0
################################# Resume ################

RESUME=exists(args.prefixLM_path)
if RESUME:
    assert os.path.exists(args.prefixLM_path), 'model file does not exist'
    loaded_obj = torch.load(args.prefixLM_path, map_location='cpu')

    PrefixLM_configure, start_epoch, weights = loaded_obj['hparams'], loaded_obj['epoch'], loaded_obj['weights']
    opt_state = loaded_obj.get('opt_state')
    scheduler_state = loaded_obj.get('scheduler_state')

############################模型#####################
tokenizer=ChineseTokenizer()
if not RESUME:
     PrefixLM_configure=dict(d_model=args.d_model,
                        input_resolution=args.input_resolution,
                        patch_size=args.patch_size,
                        num_text_tokens=tokenizer.vocab_size,
                        txt_seq_len=args.txt_seq_len,
                        heads=args.heads,
                        enc_depth=args.enc_depth,
                        dec_depth=args.dec_depth,
                        res_depth=args.res_depth,
                        d_ff=args.d_ff,
                        dropout=args.dropout)

model=PrefixLM(**PrefixLM_configure)
model.cuda(args.local_rank)
print('模型初始化完成')
model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
print('BN同步完成')
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank)
print('DDP model')
optimizers = configure_optimizers(model)
optimizer = optimizers['optimizer']
lr_scheduler = optimizers['lr_scheduler']
if RESUME:
    optimizer.load_state_dict(opt_state)
    lr_scheduler.load_state_dict(scheduler_state)
print('dataset 初始化')
train_dataset = TextImageDataset(args.data_folder,
                                 text_len=args.txt_seq_len,
                                 image_size=args.input_resolution,
                                 truncate_captions=args.truncate_captions,
                                 tokenizer=tokenizer,
                                 shuffle=args.shuffle
                                 )
print('loading dataset is complete!')
train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
print('正在同步')
synchronize()
print('dataloader 初始化')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True
                                           )

def save_model(epoch,dir,fname=None):
    save_obj = {
        'hparams': PrefixLM_configure,
        'epoch': epoch,
        'weights': model.module.state_dict(),
        'opt_state': optimizer.state_dict(),
        'scheduler_state':lr_scheduler.state_dict()
    }
    if fname is None:
        path=os.path.join(dir,'SimVLP'+str(epoch)+'.pt')
    else:
        path=os.path.join(dir,fname)
    torch.save(save_obj, path)

model.train()
for epoch in range(start_epoch,args.num_epochs):
    train_one_epoch(epoch,model,train_loader,train_sampler,optimizer,lr_scheduler,args.save_every_n_steps,rank)
if rank == 0:
   if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
   save_model(args,args.save_dir,'prefixLM.pt')
dist.destroy_process_group()#销毁进程组
