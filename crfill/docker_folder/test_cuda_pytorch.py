import torch
print(torch.cuda.is_available())
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
for i in available_gpus:
    print(i)
f = open("out.txt", "a")
f.write(torch.cuda.is_available())
f.close()
