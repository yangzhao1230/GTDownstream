import argparse
import random
import modeling
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from model.contrastive_gin import GINSimclr
from data_provider.KVDataset import KVDataset
import torch_geometric
import sys
from transformers import BertForSequenceClassification, AutoTokenizer,AutoModel,BertForPreTraining, BertConfig, BertModel

class OldModel(nn.Module):
    def __init__(self, pt_model):
        super(OldModel, self).__init__()
        self.ptmodel = pt_model
        self.emb = nn.Embedding(390, 768)

    def forward(self, input_ids, attention_mask, token_type_ids):
        embs = self.ptmodel.bert.embeddings.word_embeddings(input_ids)
        msk = torch.where(input_ids>=30700)
        for k in range(msk[0].shape[0]):
            i = msk[0][k].item()
            j = msk[1][k].item()
            embs[i,j] = self.emb(input_ids[i,j]-30700)

        return self.ptmodel.bert(inputs_embeds=embs, attention_mask=attention_mask, token_type_ids=token_type_ids)



class BigModel(nn.Module):
    def __init__(self, main_model, config):
        super(BigModel, self).__init__()
        self.main_model = main_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, tok, att):
        typ = torch.zeros(tok.shape).long().cuda()
        pooled_output = self.main_model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())['pooler_output']
        logits = self.dropout(pooled_output)
        return logits#_smi.mm(logits_des.t())

def prepare_model(args, device):

    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    
    

    bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
    bert_model = OldModel(bert_model0)
    model = BigModel(bert_model, config)
    if args.init_checkpoint is not None:
        if args.init_checkpoint=='BERT':
            con = BertConfig(vocab_size=31090,)
            bert_model = BertModel(con)
            model = BigModel(bert_model, config)
        else:
            pt = torch.load(args.init_checkpoint)
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in pt:
                pretrained_dict = {k[7:]: v for k, v in pt.items()}
                args.tok = 0
                bert_model.load_state_dict(pretrained_dict, strict=True)
                model = BigModel(bert_model, config)
            elif 'bert.embeddings.word_embeddings.weight' in pt:
                args.tok = 1
                bert_model0.load_state_dict(pt, strict=True)
                model = BigModel(bert_model0.bert, config)
    else:
        args.tok = 1
        model = BigModel(bert_model0.bert, config)

    model.to(device)

    return model

def Eval(model, dataloader, device):

    model.eval()
    with torch.no_grad():
        acc = 0
        #acc2 = 0
        allcnt = 0
       
        for batch in tqdm(dataloader):
            
            smiles, mask_smiles, text, mask_text = batch
            #aug.to(device)
            smiles = smiles.cuda()
            mask_smiles = mask_smiles.cuda()
            text = text.cuda()
            mask_text = mask_text.cuda()
            # text.to(device)
            # mask.to(device)
            #graph_rep, _ = model.graph_encoder(aug.x, aug.edge_index, aug.batch)
            #graph_rep = model.graph_proj_head(graph_rep)

            logits_smi = model(smiles, mask_smiles)
            logits_des = model(text, mask_text)

            scores = torch.cosine_similarity(logits_smi.unsqueeze(1).expand(logits_smi.shape[0], logits_smi.shape[0], logits_smi.shape[1]), logits_des.unsqueeze(0).expand(logits_des.shape[0], logits_des.shape[0], logits_des.shape[1]), dim=-1)
            argm = torch.argmax(scores, axis=1)#1
            acc += sum((argm==torch.arange(argm.shape[0]).cuda()).int()).item()
            allcnt += argm.shape[0]
            
    #print(acc/allcnt)       
    return acc/allcnt

def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    print(args)

    model = prepare_model(args, device)

    TestSet = KVDataset(args, args.data_pth + '/')

    test_dataloader = torch_geometric.loader.DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=True, drop_last=False)#True

    acc = Eval(model, test_dataloader, device)

    print('Test Acc:', acc)
    #print('Test Acc2:', acc2)
    #print('Test Acc(Text2Graph):', acc2)

  

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--config_file", default='bert_base_config.json', type=str,)
    parser.add_argument("--tok", default=0, type=int,)
    parser.add_argument("--init_checkpoint", default="/hy-tmp/origin/KV-PLM/finetune_save/ckpt_retriev03.pt", type=str,)
    parser.add_argument("--data_pth", default="/hy-tmp/GTDownstream/data", type=str,)
    parser.add_argument("--batch_size", default=64, type=int,)
    parser.add_argument("--seed", default=99, type=int,)#73 99 108
    parser.add_argument("--graph_aug", default='dnodes', type=str,)
    parser.add_argument("--text_max_len", default=128, type=int,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
