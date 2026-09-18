import sys, torch
c = torch.version.cuda or ""
print("torch", torch.__version__, "cuda", c)
sys.exit(0 if c >= "12.8" else 1)
