import argparse
from tabulate import tabulate
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary

from models.My_Vit import My_ViT
from dataload import Load_data, NYUDepthV2Dataset

def print_parameters(args):
    """
    Prints the initialization parameters in a tabular format using the logger.
    """
    table_data = [
        ["Parameter", "Value"],
        ["Input Dir Path", args.input_dir],
        ["Output Dir Path", args.output_dir],
        ["Model", args.model],
        # ["Number of Patch", args.num_patch],
        ["Batch Size", args.batch_size]
        # ["Fill Depth", parser.fill_depth]
    ]

    table = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
    print(table)
    # args.logger.info(f"Initialization parameters:\n{table}")

def get_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation NYUV2-labeled dataset")
    parser.add_argument("--input_dir", type=str, default="/home/work/coraldl/data/nyuv2_labeled_dataset",
        help="Input path")
    parser.add_argument("--output_dir", type=str, default="output",
        help="Output path")
    parser.add_argument("--model", type=str, default="",
        help="Train target model")
    # parser.add_argument("--num_patch", type=int, default="4",
    #     help="Number of Patch")
    parser.add_argument("--batch_size", type=int, default="16",
        help="Batch Size")
    
    return parser.parse_args()

# === mIoU 관련 함수 ===
def compute_confusion_matrix(preds, labels, num_classes, ignore_index=None):
    mask = (labels >= 0) & (labels < num_classes)
    if ignore_index is not None:
        mask &= (labels != ignore_index)

    hist = torch.bincount(
        num_classes * labels[mask] + preds[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).float()
    return hist

def compute_mIoU_from_confusion_matrix(conf_matrix):
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0) - intersection
    iou = intersection / (union + 1e-7)
    mIoU = iou[~iou.isnan()].mean().item()
    return mIoU, iou

def calculate_mIoU(pred_list, label_list, num_classes, ignore_index=None):
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for p, t in zip(pred_list, label_list):
        conf_matrix += compute_confusion_matrix(p.view(-1), t.view(-1), num_classes, ignore_index)
    return compute_mIoU_from_confusion_matrix(conf_matrix)


def vit(args):
    #Hyperparameter
    OUTPUT_DIR = args.output_dir
    IN_CHANNEL = 4
    NUM_CLASS = 41
    LEARNING_RATE = 1e-4  # 학습률
    BATCH_SIZE = args.batch_size
    NUM_EPOCH = 300
    WEIGHT_DECAY = 1e-2

    rgb_train, rgb_val, depth_train, depth_val, masks_train, masks_val = Load_data(args)

    train_dataset = NYUDepthV2Dataset(rgb_train, depth_train, masks_train)
    test_dataset = NYUDepthV2Dataset(rgb_val, depth_val, masks_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 데이터 확인
    sample_rgbd, sample_label = next(iter(train_loader))
    print("SAMPLE DATA CHECK")
    print(f"="*50)
    print("RGB-D shape:", sample_rgbd.shape)
    print("Label shape:", sample_label.shape)
    print(f"Mask image dtype: {sample_label.dtype}")
    print(f"num-classes: {NUM_CLASS}")
    print(f"="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = My_ViT(in_channels = IN_CHANNEL,
                    patch_size = 16,
                    emb_dim = 1024,
                    n_heads = 8,
                    img_size = (640,480),
                    depth = 12,
                    MLP_Expansion = 4,
                    MLP_dropout = 0,
                    dropout = 0.1,
                    n_classes = NUM_CLASS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    with open(f'{OUTPUT_DIR}/_summary.txt', 'w') as f:
            # 기존 stdout을 저장
            original_stdout = sys.stdout
            
            # stdout을 파일로 변경
            sys.stdout = f
            
            # 모델 요약 출력
            summary(model, input_size=[(BATCH_SIZE, 4, 640, 480)], depth=4)
            
            # stdout을 다시 원래대로 돌림
            sys.stdout = original_stdout


    # Training loop
    train_losses = []
    val_losses = []
    train_mIoUs = []
    val_mIoUs = []

    best_val_loss = float('inf')


    for epoch in range(NUM_EPOCH):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
    
        with tqdm(total=len(train_loader), desc=f"[Train] Epoch {epoch + 1}/{NUM_EPOCH}", unit='batch') as pbar:
            for rgbd, label in train_loader:
                rgbd, label = rgbd.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(rgbd)
                
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                train_preds.extend(preds.cpu())
                train_labels.extend(label.cpu())

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_mIoU, train_class_IoUs = calculate_mIoU(train_preds, train_labels, NUM_CLASS)
        train_mIoUs.append(train_mIoU)
        print(f"[Train] Epoch {epoch+1} - Loss: {avg_train_loss:.4f}, mIoU: {train_mIoU:.4f}")

        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"[Val] Epoch {epoch + 1}/{NUM_EPOCH}", unit='batch') as pbar:
                for rgbd, label in val_loader:
                    rgbd, label = rgbd.to(device), label.to(device)
                    outputs = model(rgbd)

                    loss = criterion(outputs, label)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu())
                    val_labels.extend(label.cpu())

                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)


            
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_mIoU, val_class_IoUs = calculate_mIoU(val_preds, val_labels, NUM_CLASS)
        val_mIoUs.append(val_mIoU)
        print(f"[Val] Epoch {epoch+1} - Loss: {avg_val_loss:.4f}, mIoU: {val_mIoU:.4f}")


        # 모델 가중치 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'best_unet_model_{epoch+1}.pth'))

        # 학습 결과 시각화
        now_epochs = range(0, epoch + 1)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(now_epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(now_epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(now_epochs, np.array(train_mIoUs) * 100, 'g-', label='Train mIoU Score')
        plt.plot(now_epochs, np.array(val_mIoUs) * 100, 'y-', label='Val mIoU Score')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('mIoU Score')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'))
        plt.close()

        # Save the final model
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_unet_model.pth'))

    # Save the final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_unet_model.pth'))

    # 학습 결과 시각화
    epochs = range(1, NUM_EPOCH + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(now_epochs, np.array(train_mIoUs) * 100, 'g-', label='Train mIoU Score')
    plt.plot(now_epochs, np.array(val_mIoUs) * 100, 'y-', label='Val mIoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('mIoU Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'))
    plt.close()
    

    

if __name__ == "__main__":
    args = get_args()
    print_parameters(args)

    # output 폴더 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

        
    # print("RGB-D _ Train : ", rgbd_train.shape)
    # print("RGB-D _ Val : ", rgbd_val.shape)

    

    if args.model == "vit":
        vit(args)
        print("OK")