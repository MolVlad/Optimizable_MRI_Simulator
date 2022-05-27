# %%
import cv2
import io
import numpy as np
import pylab as plt
import torch
from torch.nn.functional import pad

import warnings

import shapes_generation
from simulator import IdealB0Generator, SliceBlochSimulator2, gyromagnetic_ratio

plt.style.use('dark_background')

warnings.filterwarnings("ignore", category=UserWarning)

# %%

n_pulses = 3
size = (16, 16)
angles = torch.arange(0, 360, step=18)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# %%
def add_zero(x):
    return pad(x, (1, 0))

def minmax_scale(x):
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1.

# %%
def get_random_shape(size=(64, 64), n_points=7, center=(0., 0.), scale=1.):
    fig, _ = plt.subplots(figsize=(10, 10), dpi=90)

    a = shapes_generation.get_random_points(n=n_points, scale=1.)
    x, y, _ = shapes_generation.get_bezier_curve(a - a.mean(0, keepdims=True), rad=0.5, edgy=0.01)
    x, y = minmax_scale(x), minmax_scale(y)
    plt.fill_between(x, np.zeros_like(y), y, facecolor='w')

    plt.xlim((-1. - center[0]) / scale, (1. - center[0]) / scale)
    plt.ylim((-1. - center[1]) / scale, (1. - center[1]) / scale)

    plt.axis('off')
    io_buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    img = torch.from_numpy(cv2.resize(img_arr[:,:,0], size)).float()
    io_buf.close()
    plt.close()
    return img

def generate_phantom_mask(n_shapes, complexity=8, size=(64, 64), center_range=(-0.4, 0.4), scale_range=(.4, .8)):
    shapes = torch.stack([torch.zeros(size)] + [
        get_random_shape(size, n_points=complexity,
            center=np.random.uniform(*center_range, 2),
            scale=np.random.uniform(*scale_range)
        ) / 255 for _ in range(n_shapes)
    ])
    mask = shapes.cumsum(0).argmax(0)
    return mask

def get_values(n_shapes, pd_range=(0, 255), t1_range=(1., 5.), t2_range=(2., 7.)):
    pd = add_zero(torch.randint(*pd_range, (n_shapes,))).float()
    t1 = add_zero(torch.tensor(np.random.uniform(*t1_range, n_shapes))).float()
    t2 = add_zero(torch.tensor(np.random.uniform(*t2_range, n_shapes))).float()
    return pd, t1, t2

def sort_classes(mask):
    order = mask.flatten().bincount().argsort(descending=True)
    return order.argsort()[mask]



# %%
field_generator = IdealB0Generator(torch.device('cpu'), angles, size, mean=190., std=4.)
B_grad_plus = field_generator()
field_generator = IdealB0Generator(torch.device('cpu'), angles + 180, size, mean=190., std=4.)
B_grad_negative = field_generator()
B0 = torch.ones_like(B_grad_plus) * 190
B0[:, :2] = 0

B1 = torch.stack([torch.ones(size) * 0.525, torch.zeros(size), torch.zeros(size)])

pulse90_time = torch.linspace(0, 0.0235, 10000)
pulse90 = torch.sin(2 * np.pi * pulse90_time * 190 * gyromagnetic_ratio)
pulse180_time = torch.linspace(0.2, 0.2235, 10000)
pulse180 = 2 * torch.sin(2 * np.pi * pulse180_time * 190 * gyromagnetic_ratio)
pulse180_v = [pulse180.clone() for _ in range(n_pulses)]


plt.plot(pulse90_time, pulse90, label='90')
plt.plot(pulse180_time - 0.2, pulse180, label='180')
plt.xlim(0, 0.001)
plt.legend()
plt.xlabel('ms')
plt.show()

# %%
rf_sequence = [
    dict(field=B1, pulse=pulse90, t=pulse90_time, desc='p90', device=0),
    *[
        dict(field=B1, pulse=pulse180_v[i], t=pulse180_time + 0.4 * i, desc=f'p180_{i}', device=0)
        for i in range(n_pulses)
    ]
]

grad_sequence = [
    [
        dict(field=B_grad_negative, t=[0.2235 + 0.4 * i, 0.2935 + 0.4 * i], desc=f'G-_{i}', device=0),
        dict(field=B_grad_plus, t=[0.33 + 0.4 * i, 0.47 + 0.4 * i], desc=f'G+_{i}', device=0),
        dict(field=B_grad_negative, t=[0.5065 + 0.4 * i, 0.5765 + 0.4 * i], desc=f'G-_{i}', device=0),
    ]
    for i in range(n_pulses)
]
grad_sequence = [v for s in grad_sequence for v in s]

read_sequence = [
    dict(t=[0.37 + 0.4 * i, 0.42 + 0.4 * i], desc='readout')
    for i in range(n_pulses)
]

# dt_pulse = 24e-6
dt_wait = dt_pulse = 32e-6
dt_wait = 12e-4

timespaces = [
    torch.arange(-0.05, 0.0, dt_wait)[:-1],       # wait
    torch.arange(0.0, 0.0235, dt_pulse)[:-1],      # pulse90
    torch.arange(0.0235, 0.2, dt_wait)[:-1],      # wait
]
add_timespaces = [
    [
        torch.arange(0.2 + 0.4 * i, 0.2235 + 0.4 * i, dt_pulse)[:-1],      # pulse180
        torch.arange(0.2235 + 0.4 * i, 0.37 + 0.4 * i, dt_wait)[:-1],     # wait
        torch.arange(0.37 + 0.4 * i, 0.42 + 0.4 * i, dt_pulse)[:-1],       # read
        torch.arange(0.42 + 0.4 * i, 0.6 + 0.4 * i, dt_wait)[:-1],        # wait
    ]
    for i in range(n_pulses)
]
add_timespaces = [v for s in add_timespaces for v in s]
timespaces = timespaces + add_timespaces
timespace = torch.cat(timespaces)
# timespace = torch.linspace(timespace[0], timespace[-1], 60000)
print(f'timesteps: {len(timespace)}')

# %%
read_times = [
    [0.31 + 0.4 * i, 0.47 + 0.4 * i]
    for i in range(n_pulses)
]

# %%

# %%
#!L
def get_art_recon(simulator, M, t1, t2, res):
    angle_step = 1
    time_step = 1
    angles = simulator.angles[::angle_step]
    B0_readout = B_grad_plus[::angle_step, :, ::1, ::1]
    IH, IW = B0_readout.shape[-2:]
    target_H = res

    SE_t = simulator.timespace\
    [(simulator.timespace - t1).abs().argmin():(simulator.timespace - t2).abs().argmin()]\
    [M[:,1].cpu().abs()[
            :,
            (simulator.timespace - t1).abs().argmin():(simulator.timespace - t2).abs().argmin()
        ].mean(0).argmax()
    ].item()

    t1 = SE_t - target_H / ((B0_readout[:, 2].max() - B0_readout[:, 2].min()) * gyromagnetic_ratio)
    t2 = SE_t + target_H / ((B0_readout[:, 2].max() - B0_readout[:, 2].min()) * gyromagnetic_ratio)

    time = simulator.timespace[(simulator.timespace - t1).abs().argmin():(simulator.timespace - t2).abs().argmin()].cpu()[::time_step]
    se_data = M[:,1][:, (simulator.timespace - t1).abs().argmin():(simulator.timespace - t2).abs().argmin()].cpu()[::angle_step, ::time_step]

    se = se_data * (-2j * np.pi * B0_readout[:, 2, IH//2, IW//2][:, None] * gyromagnetic_ratio * (time[None] - SE_t)).exp()
    baseband = torch.fft.ifftshift(torch.fft.fft(se, dim=-1), dim=-1)
    freq = torch.fft.ifftshift(torch.fft.fftfreq(se.shape[1], d=(time[-1] - time[0]) / time.shape[0]))

    baseband[:, :baseband.shape[-1]//2 - target_H] = 0
    baseband[:, baseband.shape[-1]//2 + target_H:] = 0

    baseband = baseband[:, baseband.shape[-1]//2 - target_H:baseband.shape[-1]//2 + target_H].abs()
    freq = freq[baseband.shape[-1]//2 - target_H:baseband.shape[-1]//2 + target_H]
    se_down = torch.fft.fft(torch.fft.ifftshift(baseband, dim=-1), dim=-1)
    se_down = torch.fft.ifftshift(se_down, dim=-1)
    time = torch.nn.functional.interpolate(time[None, None], size=[se_down.shape[-1]])[0,0]

    E = torch.zeros([angles.shape[0], time.shape[0], IH*IW]).cfloat()
    for i, theta in enumerate(angles):
        E[i] = torch.exp(1j * 2 * np.pi * ((-B0_readout[i][2].flatten() + B0_readout[i, 2, IH//2, IW//2]) * gyromagnetic_ratio)[None] * (time - SE_t)[:, None]).cfloat()
    E = E.to(device)

    m_old = torch.zeros((IH, IW)).view(-1, 1).cfloat()
    m_old = m_old.to(device)
    _se_data = se_down.to(device)

    for i in range(1000):
        update = (((_se_data.roll(0,0) - (E @ m_old.cfloat())[:, :, 0]) / E.norm(p=2, dim=2))[:, None] * E.transpose(1, 2).conj()).mean([0,2]).view(-1, 1)
        m_old = m_old + 5e-2 * update

    rimg = m_old.reshape(IH, IW).abs()
    return rimg

# %%
#!L
def get_simulated_images(pd, t1, t2):
    simulator = SliceBlochSimulator2(
        device=device, num_isochromats_per_voxel=4,
        isOffsetStatic=True, mstd=0, offsetHz=0,
        rho_map=pd, t1_map=t1, t2_map=t2,
        angles=angles, B0=B0,
        rf_sequence=rf_sequence,
        grad_sequence=grad_sequence,
        read_sequence=read_sequence,
        timespace=timespace
    )

    M = simulator.simulate(verbose=False)
    simulator.timespace = simulator.timespace[1:]

    vec_img = []
    vec_time = []
    for t1_time, t2_time in read_times:
        rimg = get_art_recon(simulator, M, t1_time, t2_time, res=16)
        # rimg, gt = get_fbp_recon(simulator, M, t1_time, t2_time, res=12)
        # rimg = rimg.cpu().detach()
        vec_img.append(rimg)
        vec_time.append((t2_time + t1_time) / 2)
    vec_img = torch.stack(vec_img)
    vec_time = torch.tensor(vec_time)

    A = torch.vander(vec_time.cpu(), N=2)
    b = vec_img.view(-1, np.prod(size)).cpu().detach().log()
    emap = - 1 / ((A.T @ A).pinverse() @ A.T @ b)[0].reshape(size)
    emap = torch.nan_to_num(emap, 0.0)
    emap[emap < 0.5] = 0
    emap[emap > 10.0] = 0
    return vec_img.cpu(), emap

# %%
#!L
from tqdm import trange

batch_size = 100
n_batches = 9

seed = 0

torch.manual_seed(seed)
np.random.seed(seed)


pd, t1, t2 = get_values(4)

for i in range(n_batches):
    batch = {"mask": [], "pd": [], "t1": [], "t2": [], "slices": [], "reconstruct": []}

    for _ in trange(batch_size):
        seed += 1
        torch.manual_seed(seed)
        np.random.seed(seed)

        plt.style.use('dark_background')
        mask = generate_phantom_mask(4, size=size)
        mask = sort_classes(mask)
        vec_img, emap = get_simulated_images(pd[mask], t1[mask], t2[mask])
        batch["mask"].append(mask)
        batch["pd"].append(pd)
        batch["t1"].append(t1)
        batch["t2"].append(t2)
        batch["slices"].append(vec_img)
        batch["reconstruct"].append(emap)

    torch.save(batch, f"data/new_phantoms_16_20angles/batch{i}.dat")
