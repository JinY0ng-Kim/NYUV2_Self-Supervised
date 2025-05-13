import argparse
from tabulate import tabulate

from train_unet import unet
from models.PatchEmbedding import PatchEmbedding, RGBDepthFromListDataset



def print_parameters(args):
    """
    Prints the initialization parameters in a tabular format using the logger.
    """
    table_data = [
        ["Parameter", "Value"],
        ["Input Path", args.input],
        ["Output Path", args.output],
        ["Model", args.model],
        ["txt Path", args.txt_path],
        ["Number of Patch", args.num_patch],
        ["Masking Ratio", args.mask_ratio],
        ["Batch Size", args.batch_size]
        # ["Fill Depth", parser.fill_depth]
    ]

    table = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
    print(table)
    # args.logger.info(f"Initialization parameters:\n{table}")

def get_args():
    parser = argparse.ArgumentParser(description="Self-Supervised Learning NYUV2-raw")
    parser.add_argument("--input", type=str, default="",
        help="Input path")
    parser.add_argument("--output", type=str, default="",
        help="Output path")
    parser.add_argument("--model", type=str, default="",
        help="Train target model")
    parser.add_argument("--txt_path", type=str, default="",
        help="txt Path")
    parser.add_argument("--num_patch", type=int, default="4",
        help="Number of Patch")
    parser.add_argument("--mask_ratio", type=float, default="0.75",
        help="Masking ratio")
    parser.add_argument("--batch_size", type=int, default="2",
        help="Batch Size")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print_parameters(args)

    # data = PatchEmbedding(args)
    test = RGBDepthFromListDataset(args)
    for combined in test:
        print(combined.shape)
        break
    

    if args.model == "unet":
        unet(args)
    # main()