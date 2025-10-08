import os
import glob
import argparse
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
import nevergrad as ng

from model_directional_query_od import Inpainting
from utils_train import parse_args, TrainDataset, rgb_to_y, psnr, ssim, VGGPerceptualLoss
import kornia

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
perceptual_loss = VGGPerceptualLoss().to(device)

def test_loop(net, data_loader, num_iter):
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain, norain, name, h, w in test_bar:
            rain, norain = rain.to(device), norain.to(device)
            out = torch.clamp((torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()
            # computer the metrics with Y channel and double precision
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            save_path = '{}/{}/{}'.format(args.save_path, args.data_name, name[0])
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(save_path)
            test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                    .format(num_iter, 1 if args.model_file else args.num_iter,
                                            total_psnr / count, total_ssim / count))
    return total_psnr / count, total_ssim / count


def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, (num_iter if args.model_file else num_iter // 10) + 1))
    data_frame.to_csv('{}/{}.csv'.format(args.save_path, args.data_name), index_label='Iter', float_format='%.3f')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open('{}/{}.txt'.format(args.save_path, args.data_name), 'w') as f:
            f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr, best_ssim))
        torch.save(model.state_dict(), '{}/{}.pth'.format(args.save_path, args.data_name))


def train_and_evaluate(num_blocks, num_heads, channels, lr, batch_size, expansion_factor, num_refinement, loss_weights, num_iter=2, data_path='./Blind_Omni_Wav_Net/datasets/celeb', data_path_test='./Blind_Omni_Wav_Net/datasets/celeb'):
    try:
        inp_files = glob.glob(f'{data_path}/inp/*.png')
        target_files = glob.glob(f'{data_path}/target/*.png')
        print(f"Found {len(inp_files)} inp files: {inp_files}")
        print(f"Found {len(target_files)} target files: {target_files}")
        available_images = len(inp_files)
        if available_images == 0:
            return float('inf')
        batch_size = min(batch_size, 1)  # Restrict to 1 for 5-image dataset
        length = min(batch_size * num_iter, available_images)
        if batch_size == 0 or length == 0:
            return float('inf')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        perceptual_loss = VGGPerceptualLoss().to(device)
        model = Inpainting(num_blocks, num_heads, channels, num_refinement, expansion_factor).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        train_dataset = TrainDataset(data_path, data_path_test, 'inpaint', 'train', 128, length)
        if len(train_dataset) == 0:
            return float('inf')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        if len(train_loader) == 0:
            return float('inf')
        model.train()
        for n_iter, (rain, norain, name, h, w) in enumerate(train_loader):
            if n_iter >= num_iter:
                break
            rain, norain = rain.to(device), norain.to(device)
            out = model(rain)
            ssim_loss = 1 - ssim(out, norain)
            edge_out = kornia.filters.sobel(out,  normalized=True, eps=1e-06)
            edge_gt = kornia.filters.sobel(norain, normalized=True, eps=1e-06)
            edge_loss = F.l1_loss(edge_out[0], edge_gt[0]) 
            w_l1, w_percep, w_ssim, w_edge = loss_weights
            loss = F.l1_loss(out, norain)*w_l1 + perceptual_loss(out, norain)*w_percep + ssim_loss*w_ssim + edge_loss*w_edge
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation (reuse train set for demo)
        model.eval()
        with torch.no_grad():
            try:
                rain, norain, name, h, w = next(iter(train_loader))
            except StopIteration:
                return float('inf')
            rain, norain = rain.to(device), norain.to(device)
            out = model(rain)
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            val_psnr = psnr(y, gt)
        return -val_psnr.item()
    except IndexError:
        print("IndexError: Empty dataset encountered in train_and_evaluate. Skipping trial.")
        return float('inf')

# GA for architecture
instrum_ga = ng.p.Instrumentation(
    num_blocks=ng.p.Choice([[4,6,6,8], [2,4,4,6]]),
    num_heads=ng.p.Choice([[1,2,4,8], [2,2,4,8]]),
    channels=ng.p.Choice([[16,32,64,128], [24,48,96,192]]),
    num_refinement=ng.p.Scalar(lower=2, upper=6).set_integer_casting()
)
# PSO for loss weights
instrum_pso = ng.p.Instrumentation(
    w_l1=ng.p.Scalar(lower=0.1, upper=1.0),
    w_percep=ng.p.Scalar(lower=0.1, upper=0.7),
    w_ssim=ng.p.Scalar(lower=0.1, upper=0.8),
    w_edge=ng.p.Scalar(lower=0.1, upper=0.8)
)
# DE for expansion factor
instrum_de = ng.p.Instrumentation(
    expansion_factor=ng.p.Scalar(lower=2.0, upper=3.0)
)
# BO for learning rate and batch size
import optuna

def bo_objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = 1  # Restrict to 1 for 5-image dataset
    num_iter = 2
    best_ga = ga_optimizer.provide_recommendation().value
    best_pso = pso_optimizer.provide_recommendation().value
    best_de = de_optimizer.provide_recommendation().value
    # Robust extraction
    if best_ga and isinstance(best_ga, tuple) and len(best_ga) > 1 and isinstance(best_ga[1], dict):
        params_ga = best_ga[1]
        num_blocks = params_ga['num_blocks']
        num_heads = params_ga['num_heads']
        channels = params_ga['channels']
        num_refinement = params_ga['num_refinement']
    else:
        num_blocks = [4,6,6,8]
        num_heads = [1,2,4,8]
        channels = [16,32,64,128]
        num_refinement = 4
    if best_pso and isinstance(best_pso, tuple) and len(best_pso) > 1 and isinstance(best_pso[1], dict):
        params_pso = best_pso[1]
        loss_weights = (
            params_pso['w_l1'],
            params_pso['w_percep'],
            params_pso['w_ssim'],
            params_pso['w_edge']
        )
    else:
        loss_weights = (0.9, 0.5, 0.5, 0.4)
    if best_de and isinstance(best_de, tuple) and len(best_de) > 1 and isinstance(best_de[1], dict):
        params_de = best_de[1]
        expansion_factor = params_de['expansion_factor']
    else:
        expansion_factor = 2.66
    return train_and_evaluate(num_blocks, num_heads, channels, lr, batch_size, expansion_factor, num_refinement, loss_weights, num_iter=num_iter, data_path='./Blind_Omni_Wav_Net/datasets/celeb', data_path_test='./Blind_Omni_Wav_Net/datasets/celeb')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize', action='store_true', help='Run metaheuristic optimization instead of default training')
    args, unknown = parser.parse_known_args()
    if args.optimize:
        # Run metaheuristic optimization only
        # GA
        ga_optimizer = ng.optimizers.NGOpt(parametrization=instrum_ga, budget=3)
        for _ in range(3):
            try:
                x = ga_optimizer.ask()
                print("GA candidate:", x.value)
                if not isinstance(x.value, tuple) or len(x.value) < 2 or not isinstance(x.value[1], dict):
                    print("Skipping invalid GA candidate.")
                    ga_optimizer.tell(x, float('inf'))
                    continue
                params = x.value[1]
                value = train_and_evaluate(
                    params['num_blocks'],
                    params['num_heads'],
                    params['channels'],
                    1e-3,  # lr
                    1,     # batch_size
                    2.66,  # expansion_factor
                    params['num_refinement'],
                    (0.9, 0.5, 0.5, 0.4),
                    num_iter=2,
                    data_path='./Blind_Omni_Wav_Net/datasets/celeb',
                    data_path_test='./Blind_Omni_Wav_Net/datasets/celeb'
                )
                ga_optimizer.tell(x, value)
            except IndexError:
                print("IndexError in GA trial. Skipping.")
                ga_optimizer.tell(x, float('inf'))
        # PSO
        pso_optimizer = ng.optimizers.PSO(parametrization=instrum_pso, budget=3)
        for _ in range(3):
            try:
                x = pso_optimizer.ask()
                print("PSO candidate:", x.value)
                if not isinstance(x.value, tuple) or len(x.value) < 2 or not isinstance(x.value[1], dict):
                    print("Skipping invalid PSO candidate.")
                    pso_optimizer.tell(x, float('inf'))
                    continue
                params = x.value[1]
                value = train_and_evaluate(
                    [4,6,6,8],
                    [1,2,4,8],
                    [16,32,64,128],
                    1e-3,  # lr
                    1,     # batch_size
                    2.66,  # expansion_factor
                    4,     # num_refinement
                    (params['w_l1'], params['w_percep'], params['w_ssim'], params['w_edge']),
                    num_iter=2,
                    data_path='./Blind_Omni_Wav_Net/datasets/celeb',
                    data_path_test='./Blind_Omni_Wav_Net/datasets/celeb'
                )
                pso_optimizer.tell(x, value)
            except IndexError:
                print("IndexError in PSO trial. Skipping.")
                pso_optimizer.tell(x, float('inf'))
        # DE
        de_optimizer = ng.optimizers.DE(parametrization=instrum_de, budget=3)
        for _ in range(3):
            try:
                x = de_optimizer.ask()
                print("DE candidate:", x.value)
                if not isinstance(x.value, tuple) or len(x.value) < 2 or not isinstance(x.value[1], dict):
                    print("Skipping invalid DE candidate.")
                    de_optimizer.tell(x, float('inf'))
                    continue
                params = x.value[1]
                value = train_and_evaluate(
                    [4,6,6,8],
                    [1,2,4,8],
                    [16,32,64,128],
                    1e-3,  # lr
                    1,     # batch_size
                    params['expansion_factor'],
                    4,     # num_refinement
                    (0.9, 0.5, 0.5, 0.4),
                    num_iter=2,
                    data_path='./Blind_Omni_Wav_Net/datasets/celeb',
                    data_path_test='./Blind_Omni_Wav_Net/datasets/celeb'
                )
                de_optimizer.tell(x, value)
            except IndexError:
                print("IndexError in DE trial. Skipping.")
                de_optimizer.tell(x, float('inf'))
        # BO
        study = optuna.create_study(direction='minimize')
        study.optimize(bo_objective, n_trials=2)
        print('Best trial:', study.best_trial.params)

        # === Train with best found parameters and generate outputs ===
        best_params = study.best_trial.params
        # Robustly extract best GA/PSO/DE params
        best_ga = ga_optimizer.provide_recommendation().value
        best_pso = pso_optimizer.provide_recommendation().value
        best_de = de_optimizer.provide_recommendation().value
        print("best_ga:", best_ga)
        print("best_pso:", best_pso)
        print("best_de:", best_de)
        # GA
        if best_ga and isinstance(best_ga, tuple) and len(best_ga) > 1 and isinstance(best_ga[1], dict):
            params_ga = best_ga[1]
            num_blocks = params_ga['num_blocks']
            num_heads = params_ga['num_heads']
            channels = params_ga['channels']
            num_refinement = params_ga['num_refinement']
        else:
            print("Warning: best_ga is not valid, using defaults.")
            num_blocks = [4,6,6,8]
            num_heads = [1,2,4,8]
            channels = [16,32,64,128]
            num_refinement = 4
        # PSO
        if best_pso and isinstance(best_pso, tuple) and len(best_pso) > 1 and isinstance(best_pso[1], dict):
            params_pso = best_pso[1]
            loss_weights = (
                params_pso['w_l1'],
                params_pso['w_percep'],
                params_pso['w_ssim'],
                params_pso['w_edge']
            )
        else:
            print("Warning: best_pso is not valid, using defaults.")
            loss_weights = (0.9, 0.5, 0.5, 0.4)
        # DE
        if best_de and isinstance(best_de, tuple) and len(best_de) > 1 and isinstance(best_de[1], dict):
            params_de = best_de[1]
            expansion_factor = params_de['expansion_factor']
        else:
            print("Warning: best_de is not valid, using default expansion_factor.")
            expansion_factor = 2.66
        lr = best_params.get('lr', 1e-3)
        batch_size = best_params.get('batch_size', 1)
        num_iter = 500  # set to 100 for longer training
        # Use args for data_path, data_path_test, num_iter, save_path
        data_path = args.data_path if hasattr(args, 'data_path') else './Blind_Omni_Wav_Net/datasets/celeb'
        data_path_test = args.data_path_test if hasattr(args, 'data_path_test') else './Blind_Omni_Wav_Net/datasets/celeb'
        save_path = args.save_path if hasattr(args, 'save_path') else './Blind_Omni_Wav_Net/results'
        # Pass these to train_and_evaluate and to the final training loop
        # In bo_objective:
        def bo_objective(trial):
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            batch_size = 1  # Restrict to 1 for 5-image dataset
            best_ga = ga_optimizer.provide_recommendation().value
            best_pso = pso_optimizer.provide_recommendation().value
            best_de = de_optimizer.provide_recommendation().value
            # Robust extraction (same as before)
            if best_ga and isinstance(best_ga, tuple) and len(best_ga) > 1 and isinstance(best_ga[1], dict):
                params_ga = best_ga[1]
                num_blocks = params_ga['num_blocks']
                num_heads = params_ga['num_heads']
                channels = params_ga['channels']
                num_refinement = params_ga['num_refinement']
            else:
                num_blocks = [4,6,6,8]
                num_heads = [1,2,4,8]
                channels = [16,32,64,128]
                num_refinement = 4
            if best_pso and isinstance(best_pso, tuple) and len(best_pso) > 1 and isinstance(best_pso[1], dict):
                params_pso = best_pso[1]
                loss_weights = (
                    params_pso['w_l1'],
                    params_pso['w_percep'],
                    params_pso['w_ssim'],
                    params_pso['w_edge']
                )
            else:
                loss_weights = (0.9, 0.5, 0.5, 0.4)
            if best_de and isinstance(best_de, tuple) and len(best_de) > 1 and isinstance(best_de[1], dict):
                params_de = best_de[1]
                expansion_factor = params_de['expansion_factor']
            else:
                expansion_factor = 2.66
            return train_and_evaluate(num_blocks, num_heads, channels, lr, batch_size, expansion_factor, num_refinement, loss_weights, num_iter=num_iter, data_path=data_path, data_path_test=data_path_test)
        # ... rest of optimization code ...
        # In final training loop, use data_path, data_path_test, save_path, num_iter from args
        if '--optimize' in sys.argv:
            sys.argv.remove('--optimize')
        args = parse_args()
        args.num_blocks = num_blocks
        args.num_heads = num_heads
        args.channels = channels
        args.num_refinement = num_refinement
        args.expansion_factor = expansion_factor
        args.lr = lr
        args.batch_size = [batch_size]*6 if isinstance(batch_size, int) else batch_size
        args.num_iter = num_iter
        args.data_path = data_path
        args.data_path_test = data_path_test
        args.save_path = save_path
        test_dataset = TrainDataset(args.data_path_test, args.data_path_test, args.data_name, 'test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
        model = Inpainting(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor).to(device)
        print('parameters of model are',sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
        total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
        train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
        for n_iter in train_bar:
            # progressive learning
            if n_iter == 1 or n_iter - 1 in args.milestone:
                end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                start_iter = args.milestone[i - 1] if i > 0 else 0
                length = args.batch_size[i] * (end_iter - start_iter)
                train_dataset = TrainDataset(args.data_path, args.data_path_test, args.data_name, 'train', args.patch_size[i], length)
                train_loader = iter(DataLoader(train_dataset, args.batch_size[i], True, num_workers=args.workers))
                i += 1
            # train
            model.train()
            rain, norain, name, h, w = next(train_loader)
            rain, norain = rain.to(device), norain.to(device)
            out = model(rain)
            ssim_loss = 1 - ssim(out, norain)
            edge_out = kornia.filters.sobel(out,  normalized=True, eps=1e-06)
            edge_gt = kornia.filters.sobel(norain, normalized=True, eps=1e-06)
            edge_loss = F.l1_loss(edge_out[0], edge_gt[0]) 
            loss = F.l1_loss(out, norain)*loss_weights[0] + perceptual_loss(out, norain)*loss_weights[1] + ssim_loss*loss_weights[2] + edge_loss*loss_weights[3]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num += rain.size(0)
            total_loss += loss.item() * rain.size(0)
            train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                        .format(n_iter, args.num_iter, total_loss / total_num))
            lr_scheduler.step()
            if n_iter % 10 == 0:
                results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                save_loop(model, test_loader, n_iter)
            # Save inpainted image for each batch
            save_dir = os.path.join(args.save_path, args.data_name)
            os.makedirs(save_dir, exist_ok=True)
            if isinstance(name, (list, tuple)) or (hasattr(name, '__iter__') and not isinstance(name, str)):
                for img, img_name in zip(out, name):
                    img_np = img.detach().cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).clip(0, 255).astype('uint8') if img_np.max() <= 1.0 else img_np.astype('uint8')
                    out_path = os.path.join(save_dir, str(img_name))
                    print(f"Saving inpainted image to: {out_path}")
                    Image.fromarray(img_np).save(out_path)
            else:
                img = out[0]
                img_np = img.detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).clip(0, 255).astype('uint8') if img_np.max() <= 1.0 else img_np.astype('uint8')
                out_path = os.path.join(save_dir, str(name))
                print(f"Saving inpainted image to: {out_path}")
                Image.fromarray(img_np).save(out_path)
    else:
        # Run default training using command-line args
        args = parse_args()
        test_dataset = TrainDataset(args.data_path_test, args.data_path_test, args.data_name, 'test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
        model = Inpainting(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor).to(device)
        print('parameters of model are',sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))
        if args.model_file:
            model.load_state_dict(torch.load(args.model_file))
            save_loop(model, test_loader, 1)
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
            total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
            train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
            for n_iter in train_bar:
                # progressive learning
                if n_iter == 1 or n_iter - 1 in args.milestone:
                    end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                    start_iter = args.milestone[i - 1] if i > 0 else 0
                    length = args.batch_size[i] * (end_iter - start_iter)
                    train_dataset = TrainDataset(args.data_path, args.data_path_test, args.data_name, 'train', args.patch_size[i], length)
                    train_loader = iter(DataLoader(train_dataset, args.batch_size[i], True, num_workers=args.workers))
                    i += 1
                # train
                model.train()
                rain, norain, name, h, w = next(train_loader)
                rain, norain = rain.to(device), norain.to(device)
                # print(name)

                out = model(rain)

                ssim_loss = 1 - ssim(out, norain)

                edge_out = kornia.filters.sobel(out,  normalized=True, eps=1e-06)
                edge_gt = kornia.filters.sobel(norain, normalized=True, eps=1e-06)
                edge_loss = F.l1_loss(edge_out[0], edge_gt[0]) 

                loss = F.l1_loss(out, norain)*0.9 + perceptual_loss(out, norain)*0.5 + ssim_loss*0.5 + edge_loss*0.4

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_num += rain.size(0)
                total_loss += loss.item() * rain.size(0)
                train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                          .format(n_iter, args.num_iter, total_loss / total_num))

                lr_scheduler.step()
                if n_iter % 10 == 0:
                    results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                    save_loop(model, test_loader, n_iter)

                # Save inpainted image for each batch
                save_dir = os.path.join(args.save_path, args.data_name)
                os.makedirs(save_dir, exist_ok=True)
                if isinstance(name, (list, tuple)) or (hasattr(name, '__iter__') and not isinstance(name, str)):
                    for img, img_name in zip(out, name):
                        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np * 255).clip(0, 255).astype('uint8') if img_np.max() <= 1.0 else img_np.astype('uint8')
                        out_path = os.path.join(save_dir, str(img_name))
                        print(f"Saving inpainted image to: {out_path}")
                        Image.fromarray(img_np).save(out_path)
                else:
                    img = out[0]
                    img_np = img.detach().cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).clip(0, 255).astype('uint8') if img_np.max() <= 1.0 else img_np.astype('uint8')
                    out_path = os.path.join(save_dir, str(name))
                    print(f"Saving inpainted image to: {out_path}")
                    Image.fromarray(img_np).save(out_path)
