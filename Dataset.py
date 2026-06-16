  # -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

class UpscaleDataset(torch.utils.data.Dataset):
    PRECIPITATION_NAMES = [
        'rf', 'RF', 'rainfall', 'RAINFALL', 'precipitation', 'PRECIPITATION',
        'pr', 'PR', 'tp', 'TP', 'total_precipitation', 'prec', 'PREC', 'rain', 'RAIN'
    ]

    def __init__(
        self,
        nc_file,
        oro_file,
        d2m_file=None,
        downscale_factor=4,
        normalize=True,
        device="cpu", # Let DDP dataloader handle pinning
        auto_detect_var=True,
        variable_name=None,
        split='train',
    ):
        self.split = split
        self.downscale = downscale_factor
        self.has_d2m = d2m_file is not None
        self.log_transformed = normalize

        # ── LOAD DATASETS ──
        ds = xr.open_dataset(nc_file, engine="netcdf4" if nc_file.endswith('.nc') else "h5netcdf")
        ds_oro = xr.open_dataset(oro_file, engine="netcdf4")
        
        # Ensure spatial alignment
        for dataset in [ds, ds_oro]:
            lat = next((c for c in dataset.coords if c.lower() in ['lat', 'latitude']), None)
            lon = next((c for c in dataset.coords if c.lower() in ['lon', 'longitude']), None)
            if lat and lon:
                dataset = dataset.sortby([lat, lon])

        topo = np.nan_to_num(ds_oro['topology'].values.astype(np.float32), nan=0.0)

        if self.has_d2m:
            ds_d2m = xr.open_dataset(d2m_file, engine="netcdf4")
            lat_d2m = next((c for c in ds_d2m.coords if c.lower() in ['lat', 'latitude']), None)
            lon_d2m = next((c for c in ds_d2m.coords if c.lower() in ['lon', 'longitude']), None)
            if lat_d2m and lon_d2m:
                ds_d2m = ds_d2m.sortby([lat_d2m, lon_d2m])
            
            d2m_var = next((v for v in ['d2m', 'D2M', 'dewpoint_temperature_2m'] if v in ds_d2m.data_vars), list(ds_d2m.data_vars)[0])
            d2m_raw = np.nan_to_num(ds_d2m[d2m_var].values.astype(np.float32), nan=0.0)
            if d2m_raw.ndim == 4: d2m_raw = d2m_raw[:, 0]
            if d2m_raw.mean() > 100: d2m_raw -= 273.15 # Kelvin to Celsius

        # ── VARIABLE SELECTION ──
        if variable_name is not None:
            data = ds[variable_name].values
        elif auto_detect_var:
            found = next((v for v in ds.data_vars if any(k.lower() in v.lower() for k in self.PRECIPITATION_NAMES)), None)
            if not found: raise ValueError("CRITICAL: No precipitation variable detected.")
            data = ds[found].values
        else:
            data = ds[list(ds.data_vars)[0]].values

        data = np.clip(np.nan_to_num(data.astype(np.float32), nan=0.0), a_min=0.0, a_max=None)

        dt = pd.to_datetime(ds["TIME"].values)
        self.doy = torch.tensor(dt.dayofyear.values / 366.0, dtype=torch.float32)
        self.hour = torch.tensor(dt.hour.values / 24.0, dtype=torch.float32)

        # ── NORMALIZATION ──
        self.topo_mean, self.topo_std = topo.mean(), topo.std() + 1e-8
        if normalize: topo = (topo - self.topo_mean) / self.topo_std

        if self.has_d2m:
            self.d2m_mean, self.d2m_std = float(np.nanmean(d2m_raw)), float(np.nanstd(d2m_raw) + 1e-8)
            if normalize: d2m_raw = (d2m_raw - self.d2m_mean) / self.d2m_std

        # ── CROP TO MULTIPLE OF 16 ──
        H, W = (data.shape[1] // 16) * 16, (data.shape[2] // 16) * 16
        data, topo = data[:, :H, :W], topo[:H, :W]
        if self.has_d2m: d2m_raw = d2m_raw[:, :H, :W]

        self.H, self.W, T = H, W, data.shape[0]
        self.doy, self.hour = self.doy[:T], self.hour[:T]
        self.topo_tensor = torch.from_numpy(topo).unsqueeze(0).contiguous()
        if self.has_d2m: self.d2m_tensor = torch.from_numpy(d2m_raw).unsqueeze(1).contiguous()

        # ── PRECOMPUTE FINE / COARSE (STRICT CONSERVATION POOLING) ──
        fine_all, coarse_all = [], []
        # Chunking prevents RAM blowout on massive NetCDF files
        for i in range(0, T, 64):
            chunk = torch.from_numpy(data[i:i+64]).unsqueeze(1)
            # F.avg_pool2d is physically required here to conserve water mass when creating GT 100km data
            coarse = F.avg_pool2d(chunk, kernel_size=self.downscale, stride=self.downscale)
            if self.log_transformed:
                chunk, coarse = torch.log1p(chunk), torch.log1p(coarse)
            fine_all.append(chunk)
            coarse_all.append(coarse)

        self.fine = torch.cat(fine_all).contiguous()
        self.coarse = torch.cat(coarse_all).contiguous()

        # ── VARIANCE MAP (STAYS COARSE) ──
        var_map = np.nan_to_num(np.var(self.coarse[:, 0].numpy(), axis=0), nan=0.0)
        vmin, vmax = var_map.min(), var_map.max()
        var_map = (var_map - vmin) / (vmax - vmin + 1e-8) if vmax - vmin > 1e-7 else np.zeros_like(var_map)
        self.var_map_tensor = torch.from_numpy(var_map).float().unsqueeze(0).contiguous()

        print(f"[Dataset] Split: {self.split} | Fine (25km): {self.H}x{self.W} | Coarse (100km): {self.H//self.downscale}x{self.W//self.downscale}")

    def __len__(self):
        return self.fine.shape[0]

    def __getitem__(self, idx):
        T_COND = 5 
        start_idx = idx - T_COND + 1
        
        if start_idx < 0:
            padding = self.coarse[0:1].repeat(abs(start_idx), 1, 1, 1)
            actual_frames = self.coarse[0 : idx + 1]
            tc_frames = torch.cat([padding, actual_frames], dim=0).squeeze(1)
        else:
            tc_frames = self.coarse[start_idx : idx + 1].squeeze(1)

        # Spatial upscaling without bilinear blurring to match model input expectations
        # The network's convolutions will smooth this properly.
        tc_up = tc_frames.unsqueeze(1).repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1).squeeze(1)

        out = {
            "fine": self.fine[idx],
            "coarse": self.coarse[idx],
            "tc_frames": tc_up,
            "topo": self.topo_tensor,
            "doy": self.doy[idx],
            "hour": self.hour[idx],
            "var_map": self.var_map_tensor,
        }
        if self.has_d2m: out["d2m"] = self.d2m_tensor[idx]
        return out

    def denormalize(self, data):
        return torch.expm1(data) if self.log_transformed else data

    def denormalize_d2m(self, data):
        return data * self.d2m_std + self.d2m_mean
