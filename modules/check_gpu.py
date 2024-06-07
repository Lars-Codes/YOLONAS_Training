import torch 
def check_gpu(): 
    if(torch.cuda.is_available()):
        return True
    else:
        return False
    
if __name__ == "__main__":
    if check_gpu():
        exit(1)
    else: 
        exit(0)
        