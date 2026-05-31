import os, glob, argparse
import numpy as np
import torch

def parse_args():
    p = argparse.ArgumentParser(description="Convert GVHMR hmr4d .pt -> PHUMA (N,69) .npy")
    p.add_argument("--input_folder",  default="data/GVHMR/dataset_turning_joe_2/pt")
    p.add_argument("--output_folder", default="data/GVHMR/dataset_turning_joe_2/npy")
    return p.parse_args()

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    for pt in sorted(glob.glob(os.path.join(args.input_folder, "*.pt"))):
        d = torch.load(pt, map_location="cpu", weights_only=False)  # weights_only=False: .pt holds non-tensors
        g = d["smpl_params_global"]
        arr = np.concatenate([
            g["transl"].cpu().numpy(),         # (N,3)
            g["global_orient"].cpu().numpy(),  # (N,3)
            g["body_pose"].cpu().numpy(),      # (N,63)
        ], axis=1).astype(np.float32)          # -> (N,69)
        name = os.path.splitext(os.path.basename(pt))[0]
        out = os.path.join(args.output_folder, name + ".npy")
        np.save(out, arr)
        print(f"{name}: {arr.shape} -> {out}")

if __name__ == "__main__":
    main(parse_args())