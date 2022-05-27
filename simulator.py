import math
import numpy as np
import pylab as plt
import torch
import h5py
import logging
from tqdm import tqdm
from typing import List, Dict
gyromagnetic_ratio = 42.58 # kHz / mT


class IdealB0Generator:

    def __init__(
            self, device: torch.device,
            angles: torch.tensor, img_shape: List[int],
            mean: float, std: float
    ):
        self.device = device
        self.angles = angles
        self.img_shape = img_shape
        self.mean = mean
        self.std = std

    @staticmethod
    def getB(theta, shape, mean, std, device):
        x = torch.arange(shape[0], device=device)[None] / shape[0]
        y = torch.arange(shape[1], device=device)[:, None] / shape[1]
        B = x * torch.sin(theta) + y * torch.cos(theta)
        B = (B - B.mean()) / (B.std() + 1e-11)
        return torch.stack([torch.zeros_like(B), torch.zeros_like(B), B * std + mean])

    def __call__(self, disable_verbose=False):
        result = []
        for angle in tqdm(self.angles, disable=disable_verbose):
            result.append(self.getB(angle.deg2rad(), self.img_shape, self.mean, self.std, self.device))
        return torch.stack(result)


class SliceBlochSimulator:

    def __init__(
            self, device: torch.device,
            num_isochromats_per_voxel: int, img_phantom: torch.tensor,
            B0_rf_mT: torch.tensor, B0_read_mT: torch.tensor, angles: torch.tensor,
            B1_mT: torch.tensor,
            pulse90_timeMs: float, pulse180_timeMs: float,
            readout_timeMs: float, T1ms: float, T2ms: float,
            pulse90_sequence: torch.tensor,
            pulse90_sequence_timeMs: torch.tensor,
            pulse180_sequence: torch.tensor,
            pulse180_sequence_timeMs: torch.tensor,
            timeMs0: float, timeMs1: float, timesteps: int,
            mstd: float, offsetHz: float, isOffsetStatic: bool,
    ):
        self.device = device
        assert len(img_phantom.shape) == 2, f'img_phantom must be 2d, but got {img_phantom.shape}'
        assert len(B0_rf_mT.shape) == 4, f'B0_rf_mT must be 3d: Nx3xHxW, but got {B0_rf_mT.shape}'
        assert len(B0_read_mT.shape) == 4, f'B0_read_mT must be 3d: Nx3xHxW, but got {B0_read_mT.shape}'
        assert len(B1_mT.shape) == 3, f'B1_mT must be 3d: 3xHxW, but got {B1_mT.shape}'
        assert B0_rf_mT.shape == B0_read_mT.shape
        assert B0_rf_mT.shape[1:] == B1_mT.shape
        assert B1_mT.shape[1:] == img_phantom.shape
        assert timeMs0 < timeMs1
        assert len(angles.shape) == 1
        self.isOffsetStatic = isOffsetStatic
        self.num_steps = timesteps
        self.img_phantom = img_phantom.to(device)
        self.num_isochromats_per_voxel = num_isochromats_per_voxel
        num_rotations, _, H, W = B0_rf_mT.shape
        assert angles.shape[0] == num_rotations
        self.angles = angles.to(device)
        self.num_rotations = num_rotations
        self.total_isochromats = num_rotations * H * W * num_isochromats_per_voxel
        self.timespace = torch.linspace(timeMs0, timeMs1, self.num_steps, device=device)
        self.dt = self.timespace[1] - self.timespace[0]
        self.M = torch.randn(3, self.total_isochromats, device=self.device) * mstd
        self.M += torch.eye(3)[2].to(device)[:, None]
        self.M *= self.img_phantom[None, None]\
            .repeat(num_isochromats_per_voxel * len(angles),1,1,1)\
            .movedim(0,-1)\
            .flatten(1)
        self.b_isochromat_std = (offsetHz * 1e-3 / gyromagnetic_ratio)
        self.B_background = torch.randn(3, self.total_isochromats, device=self.device) * self.b_isochromat_std
        self.B0_rf = B0_rf_mT.repeat(num_isochromats_per_voxel, 1, 1, 1).movedim(0, -1).flatten(1).to(device)
        self.B0_read = B0_read_mT.repeat(num_isochromats_per_voxel, 1, 1, 1).movedim(0, -1).flatten(1).to(device)
        self.B1 = B1_mT.repeat(num_isochromats_per_voxel * self.num_rotations,1,1,1).movedim(0,-1).flatten(1).to(device)
        self.p90_seq = pulse90_sequence.to(device)
        self.p180_seq = pulse180_sequence.to(device)
        self.p90_seq_time = pulse90_sequence_timeMs.to(device)
        self.p180_seq_time = pulse180_sequence_timeMs.to(device)
        self.p90_time = pulse90_timeMs
        self.p180_time = pulse180_timeMs
        self.read_time = readout_timeMs
        self.T1 = T1ms
        self.T2 = T2ms
        self.p90_duration = self.p90_seq_time[-1] - self.p90_seq_time[0]
        self.p180_duration = self.p180_seq_time[-1] - self.p180_seq_time[0]
        self.t = self.timespace[0]
        self.step = 0
        self.isPulse90Enabled = True
        self.isPulse180Enabled = True
        self.Mh = torch.zeros((self.num_rotations, 3, self.num_steps), dtype=torch.float32, device=self.device)

    @staticmethod
    def getR(B, dt):
        Bnorm = B.norm(dim=0)
        nx = B[0] / Bnorm
        ny = B[1] / Bnorm
        nz = B[2] / Bnorm
        phi = dt * gyromagnetic_ratio * Bnorm * 2 * math.pi
        R11 = nx ** 2 + (1 - nx ** 2) * torch.cos(phi)
        R12 = nx * ny * (1 - torch.cos(phi)) + nz * torch.sin(phi)
        R13 = nx * nz * (1 - torch.cos(phi)) - ny * torch.sin(phi)
        R21 = nx * ny * (1 - torch.cos(phi)) - nz * torch.sin(phi)
        R22 = nx ** 2 + (1 - nx ** 2) * torch.cos(phi)
        R23 = ny * nz * (1 - torch.cos(phi)) + nx * torch.sin(phi)
        R31 = nx * nz * (1 - torch.cos(phi)) + ny * torch.sin(phi)
        R32 = ny * nz * (1 - torch.cos(phi)) - nx * torch.sin(phi)
        R33 = nz ** 2 + (1 - nz ** 2) * torch.cos(phi)
        R = torch.stack([
            torch.stack([R11, R12, R13]),
            torch.stack([R21, R22, R23]),
            torch.stack([R31, R32, R33])
        ])
        return R.T

    def updateM(self, pbar: tqdm, desc=''):
        status_str = 'None'
        B = self.B_background
        if self.t - self.p90_duration < self.p90_time <= self.t and self.isPulse90Enabled:
            idx = (self.t - self.p90_seq_time - self.p90_time).abs().argmin()
            rf_value = self.p90_seq[idx]
            B = B + self.B0_rf + self.B1 * rf_value
            status_str = 'pi/2 pulse'
        elif self.t - self.p180_duration < self.p180_time <= self.t and self.isPulse180Enabled:
            idx = (self.t - self.p180_seq_time - self.p180_time).abs().argmin()
            rf_value = self.p180_seq[idx]
            B = B + self.B0_rf + self.B1 * rf_value
            status_str = 'pi pulse'
        elif self.t > self.p180_time + self.p180_duration and self.isPulse180Enabled:
            B = B + self.B0_read
            status_str = 'B0 read mode'
        elif self.t > self.read_time:
            B = B + self.B0_read
            status_str = 'B0 read mode'
        else:
            B = B + self.B0_rf
            status_str = 'B0 rf mode'
        if not self.isOffsetStatic:
            B = B + torch.randn_like(B) * self.b_isochromat_std
        self.M = torch.matmul(self.getR(-B, self.dt), self.M.T[:, :, None])[:, :, 0].T
        self.M = (self.M[0] * math.exp(-self.dt / self.T2)) * torch.eye(3)[0][:, None].to(self.device) +\
                 (self.M[1] * math.exp(-self.dt / self.T2)) * torch.eye(3)[1][:, None].to(self.device) +\
                 (self.M[2] * math.exp(-self.dt / self.T1) + (1 - math.exp(-self.dt / self.T1))) * torch.eye(3)[2][:, None].to(self.device)
        if pbar is not None:
            if self.step % 500 == 0:
                pbar.set_description(desc=f'[{status_str}] {desc} time {round(self.t.item(), 4)} msec')
        self.step += 1
        self.t = self.timespace[self.step if self.step < self.num_steps else 0]
        self.Mh[:, :, self.step - 1] = self.M.view(3, -1, self.num_rotations).mean(1).swapaxes(0, 1)

    def simulate(self, verbose: bool = True):
        pbar = tqdm(enumerate(self.timespace), total=len(self.timespace), disable=not verbose, mininterval=10, maxinterval=30)
        for i, t in pbar:
            self.updateM(pbar)
        return self.Mh


class SliceBlochSimulator2:

    def __init__(
            self, device: torch.device,
            num_isochromats_per_voxel: int, 
            rho_map: torch.tensor, t1_map: torch.tensor, t2_map: torch.tensor,
            angles: torch.tensor,
            rf_sequence: List[Dict[str, torch.tensor]],
            grad_sequence: List[Dict[str, torch.tensor]],
            read_sequence: List[Dict[str, torch.tensor]],
            timespace: torch.tensor, B0: torch.tensor,
            mstd: float, offsetHz: float, isOffsetStatic: bool,
    ):
        self.device = device
        self.B0 = B0.repeat(num_isochromats_per_voxel, 1, 1, 1).movedim(0, -1).flatten(1).to(device)
        assert rho_map.shape == t1_map.shape == t2_map.shape
        assert len(rho_map.shape) == 2, f'img_phantom must be 2d, but got {rho_map.shape}'
        assert len(angles.shape) == 1
        self.isOffsetStatic = isOffsetStatic
        self.num_isochromats_per_voxel = num_isochromats_per_voxel
        H, W = rho_map.shape
        num_rotations = angles.shape[0]
        self.rf_sequence = self.process_rf_sequences(rf_sequence, self.num_isochromats_per_voxel, num_rotations, self.device)
        self.grad_sequence = self.process_grad_sequences(grad_sequence, self.num_isochromats_per_voxel, num_rotations, self.device)
        self.read_sequence = read_sequence
        self.rho_map = rho_map.to(device)
        self.angles = angles.to(device)
        self.num_rotations = num_rotations
        self.total_isochromats = num_rotations * H * W * num_isochromats_per_voxel
        self.timespace = timespace.to(device)
        self.M = torch.randn(3, self.total_isochromats, device=self.device) * mstd
        self.M += torch.eye(3)[2].to(device)[:, None]
        self.M *= self.rho_map[None, None]\
            .repeat(num_isochromats_per_voxel * len(angles),1,1,1)\
            .movedim(0,-1)\
            .flatten(1)
        self.t1_map = t1_map.to(device)[None, None]\
            .repeat(num_isochromats_per_voxel * len(angles),1,1,1)\
            .movedim(0,-1)\
            .flatten(1)[0]
        self.t2_map = t2_map.to(device)[None, None]\
            .repeat(num_isochromats_per_voxel * len(angles),1,1,1)\
            .movedim(0,-1)\
            .flatten(1)[0]
        self.b_isochromat_std = (offsetHz * 1e-3 / gyromagnetic_ratio)
        self.B_background = torch.randn(3, self.total_isochromats, device=self.device) * self.b_isochromat_std
        self.Mh = torch.zeros((self.num_rotations, 3, self.timespace.shape[0] - 1), dtype=torch.float32)

    @staticmethod
    def process_rf_sequences(seq: List, ipv: int, nr: int, device: torch.device):
        new_seq = []
        for d in seq:
            new_seq.append(
                dict(
                    field=d['field'].repeat(ipv * nr, 1, 1, 1)\
                                    .movedim(0,-1).flatten(1).to(f"cuda:{d['device']}"),
                    pulse=d['pulse'].to(f"cuda:{d['device']}"),
                    t=d['t'],
                    desc=d['desc']
                )
            )
        return new_seq

    @staticmethod
    def process_grad_sequences(seq: List, ipv: int, nr: int, device: torch.device):
        new_seq = []
        for d in seq:
            new_seq.append(
                dict(
                    field=d['field'].repeat(ipv, 1, 1, 1)\
                                    .movedim(0,-1).flatten(1).to(f"cuda:{d['device']}"),
                    t=d['t'],
                    desc=d['desc']
                )
            )
        return new_seq

    @staticmethod
    def getR(B, dt):
        Bnorm = B.norm(dim=0)
        nx = B[0] / Bnorm
        ny = B[1] / Bnorm
        nz = B[2] / Bnorm
        phi = dt * gyromagnetic_ratio * Bnorm * 2 * math.pi
        R11 = nx ** 2 + (1 - nx ** 2) * torch.cos(phi)
        R12 = nx * ny * (1 - torch.cos(phi)) + nz * torch.sin(phi)
        R13 = nx * nz * (1 - torch.cos(phi)) - ny * torch.sin(phi)
        R21 = nx * ny * (1 - torch.cos(phi)) - nz * torch.sin(phi)
        R22 = nx ** 2 + (1 - nx ** 2) * torch.cos(phi)
        R23 = ny * nz * (1 - torch.cos(phi)) + nx * torch.sin(phi)
        R31 = nx * nz * (1 - torch.cos(phi)) + ny * torch.sin(phi)
        R32 = ny * nz * (1 - torch.cos(phi)) - nx * torch.sin(phi)
        R33 = nz ** 2 + (1 - nz ** 2) * torch.cos(phi)
        R = torch.stack([
            torch.stack([R11, R12, R13]),
            torch.stack([R21, R22, R23]),
            torch.stack([R31, R32, R33])
        ])
        return R.T

    def simulate(self, verbose: bool = True):
        pbar = tqdm(enumerate(self.timespace[1:]), total=len(self.timespace[1:]), disable=not verbose, mininterval=2, maxinterval=30)
        queued_rf = self.rf_sequence.pop(0) if self.rf_sequence else None
        queued_grad = self.grad_sequence.pop(0) if self.grad_sequence else None
        queued_read = self.read_sequence.pop(0) if self.read_sequence else None
        status = ''
        for i, t in pbar:
            # Update queued sequences
            if queued_rf and queued_rf['t'][-1] < t.item():
                queued_rf = self.rf_sequence.pop(0) if self.rf_sequence else None
            if queued_grad and queued_grad['t'][-1] < t.item():
                queued_grad = self.grad_sequence.pop(0) if self.grad_sequence else None
            if queued_read and queued_read['t'][-1] < t.item():
                queued_read = self.read_sequence.pop(0) if self.read_sequence else None
            # get current device
            if queued_grad:
                current_device = queued_grad['field'].device
            self.B_background = self.B_background.to(current_device)
            self.B0 = self.B0.to(current_device)
            self.M = self.M.to(current_device)
            self.t1_map = self.t1_map.to(current_device)
            self.t2_map = self.t2_map.to(current_device)
            self.rho_map = self.rho_map.to(current_device)
            # Init B superposition
            B = self.B_background
            _status = []
            # Apply RF?
            if queued_rf and queued_rf['t'][0] < t.item() <= queued_rf['t'][-1]:
                t_idx = (t.item() - queued_rf['t']).abs().argmin()
                B = B + queued_rf['field'] * queued_rf['pulse'][t_idx]
                _status += [queued_rf['desc']]
            # Apply gradient?
            if queued_grad and queued_grad['t'][0] < t.item() <= queued_grad['t'][-1]:
                B = B + queued_grad['field']
                _status += [queued_grad['desc']]
            else:
                B = B + self.B0
                _status += ['B0']
            # Do read?
            if queued_read and queued_read['t'][0] < t.item() <= queued_read['t'][-1]:
                _status += [queued_read['desc']]
            if not self.isOffsetStatic:
                B = B + torch.randn_like(B) * self.b_isochromat_std
            if status != ' '.join(_status):
                status = ' '.join(_status)
                pbar.set_description(desc=status)
            dt = t - self.timespace[i]
            dt = dt.to(current_device)
            self.M = torch.matmul(self.getR(-B, dt), self.M.T[:, :, None])[:, :, 0].T
            self.M = (self.M[0] * torch.exp(-dt / self.t2_map)) * torch.eye(3, device=current_device)[0][:, None] +\
                     (self.M[1] * torch.exp(-dt / self.t2_map)) * torch.eye(3, device=current_device)[1][:, None] +\
                     (self.M[2] * torch.exp(-dt / self.t1_map) + (1 - torch.exp(-dt / self.t1_map))) * torch.eye(3, device=current_device)[2][:, None]  
            self.Mh[:, :, i] = self.M.view(3, -1, self.num_rotations).mean(1).swapaxes(0, 1).cpu()
        return self.Mh
