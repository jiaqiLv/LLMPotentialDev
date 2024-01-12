import torch
from zmq import device
from utils import *
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from torch.nn.functional import cross_entropy
import torch

def cal_cross_attention(model_path,file1,file2,is_str=False,device_map = "auto"):
    if not is_str:
        file1 = file2str(file1)
        file2 = file2str(file2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype= torch.bfloat16,
    #     device_map = device_map
    # )
    # model = model.eval()
    file1_token = tokenizer(file1, return_tensors='pt')
    file2_token = tokenizer(file2, return_tensors='pt')
    file1_ids = file1_token['input_ids'][0].float()
    file2_ids = file2_token['input_ids'][0].float()
    diff = len(file1_token['input_ids'][0]) - len(file2_token['input_ids'][0])
    if diff<0:
        file1_ids = torch.nn.functional.pad(file1_ids,(0,abs(diff)),value=0)
    elif diff>0:
        file2_ids = torch.nn.functional.pad(file2_ids,(0,abs(diff)),value=0)
    print(file1_ids)
    print(len(file1_ids))
    print(len(file2_ids))

    loss = cross_entropy(file1_ids,file2_ids)
    print("Cross Entropy Loss:", loss.item())

def test_case_cal_cross_attention():
    file1_path = '/code/LLMPotentialDev/RAG4CodeComplement/datasets/operator/cpu/gt/flatten[7__4__20__20]_1_gt.c'
    file2_path = '/code/LLMPotentialDev/RAG4CodeComplement/datasets/operator/cpu/generate/flatten[7__4__20__20]_1_generate.c'
    model_path = '/code/LLM4HPCTransCompile/qlora/save_model/CodeLlama-34b-Instruct-hf-100-v2.0_topi_with_ir'
    cal_cross_attention(model_path,file2_path,file1_path)

if __name__ == "__main__":
    test_case_cal_cross_attention()