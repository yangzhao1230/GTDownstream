import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from model.contrastive_gin import GINSimclr
from data_provider.match_dataset import GINMatchDataset
import torch_geometric

def prepare_model(args, device):

    model = GINSimclr.load_from_checkpoint(args.init_checkpoint)
    model.to(device)

    return model

def Eval(model, dataloader, device):

    model.eval()
    with torch.no_grad():
        acc1 = 0
        acc2 = 0
        allcnt = 0
       
        for batch in tqdm(dataloader):
            
            aug, text, mask = batch
            aug.to(device)
            text = text.cuda()
            mask = mask.cuda()
            # text.to(device)
            # mask.to(device)
            graph_rep, _ = model.graph_encoder(aug.x, aug.edge_index, aug.batch)
            graph_rep = model.graph_proj_head(graph_rep)

            text_rep = model.text_encoder(text, mask)
            text_rep = model.text_proj_head(text_rep)

            scores1 = torch.cosine_similarity(graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            scores2 = torch.cosine_similarity(text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)

            argm1 = torch.argmax(scores1, axis=1)
            argm2 = torch.argmax(scores2, axis=1)

            acc1 += sum((argm1==torch.arange(argm1.shape[0]).cuda()).int()).item()
            acc2 += sum((argm2==torch.arange(argm2.shape[0]).cuda()).int()).item()

            allcnt += argm1.shape[0]
            
    #print(acc/allcnt)       
    return acc1/allcnt, acc2/allcnt

def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    print(args)

    model = prepare_model(args, device)

    TestSet = GINMatchDataset(args, args.data_pth + '/')

    test_dataloader = torch_geometric.loader.DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=True, drop_last=False)#True

    acc1, acc2 = Eval(model, test_dataloader, device)

    print('Test Acc:', acc1)
    #print('Test Acc(Text2Graph):', acc2)

  

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--init_checkpoint", default="../KV-PLM/save_model/model.ckpt", type=str,)
    parser.add_argument("--data_pth", default="../KV-PLM/data/train", type=str,)
    parser.add_argument("--batch_size", default=16, type=int,)
    parser.add_argument("--seed", default=99, type=int,)#73 99 108
    parser.add_argument("--graph_aug", default='dnodes', type=str,)
    parser.add_argument("--text_max_len", default=128, type=int,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
