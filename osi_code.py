import pandas as pd
import numpy as np
import scipy.linalg as la
import scipy.signal as signal
import matplotlib.pyplot as plt
import psycopg2
import time
import re
import openpyxl
from psycopg2 import sql
from scipy import stats
import datetime

print("=== Detecting ===")


def load_data(file_name, prefix):
    print(f'Loading {prefix.upper()}.....')
    df = pd.read_excel(file_name, skiprows=0, nrows=30000)
    df = df.iloc[:, 1:]
    var_list = []
    for i, col in enumerate(df.columns, start=1):
        var_name = f"{prefix}{i}"
        data = df[[col]].transpose().interpolate(method="linear", axis=1)
        globals()[var_name] = data
        var_list.append(data)
    return var_list
try:
    P_all = load_data("Data_Raw_P.xlsx", "p")
    
    df = pd.read_excel("Data_Raw_P.xlsx")
    cols = df.keys()
    print("Loaded P Data")
    
    data_except_first_col = df.iloc[:, 1:]
    if data_except_first_col.isna().all().all():
        print("Data is empty (only first column has data). Restarting...\n")
        time.sleep(1)
        exit()
    
    if 'p1' not in globals():
        print("Error: p1 not found. Check your Excel file columns.")
        exit()
    #%%
    t = np.arange(0, round((p1.shape[1]-1)*0.04, 2) + 0.04, 0.04)
    data = p2.values.reshape(-1, 1)
    data = data.squeeze()
    
    window = 20
    n = len(data)
    smoothed_data = np.empty(n)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + (window + 1) // 2)
        smoothed_data[i] = np.mean(data[start:end])
    data = smoothed_data
    
    Fs = 1 / (t[1] - t[0])
    ambient_duration = 5
    Nw = 2 * Fs
    threshold_multiplier = 1.5
    consecutive_thresh = 10
    
    ambient_samples = int(np.round(ambient_duration * Fs))
    ambient_data = data[:ambient_samples]
    mu_ambient = np.mean(ambient_data)
    sigma_ambient = np.std(ambient_data, ddof=1)
    eps = np.finfo(float).eps
    z_scores = (data - mu_ambient) / (sigma_ambient + eps)
    
    ambient_energy = np.mean(np.sum(z_scores[:ambient_samples]**2, axis=0))
    ambient_l2norm = np.sqrt(ambient_energy)
    
    energy = np.zeros_like(data)
    l2norm = np.zeros_like(data)
    Nw = int(Nw)
    
    for i in range(Nw - 1, len(data)):
        window = z_scores[i - Nw + 1 : i + 1]
        energy[i] = np.sum(window**2, axis=0)
        l2norm[i] = np.sqrt(energy[i])
    
    energy_threshold = threshold_multiplier * ambient_energy
    l2norm_threshold = threshold_multiplier * ambient_l2norm
    
    detection = energy > energy_threshold
    validated_detection = np.zeros_like(detection, dtype=int)
    consecutive_count = 0
    event_start_idx = -1
    
    for i in range(len(detection)):
        if detection[i]:
            consecutive_count += 1
            if consecutive_count == consecutive_thresh:
                event_start_idx = i
        else:
            consecutive_count = 0
            continue
        if event_start_idx != -1 and i >= event_start_idx:
            validated_detection[i] = 1
    
    detection_time = t[event_start_idx] if event_start_idx > 0 else None
    t_range = t[1] - t[0]
    detect_idx = int(detection_time/t_range) if detection_time else 0
    t_series = pd.read_excel('Data_Raw_P.xlsx', usecols='A', skiprows=0)
    t_timestamps = t_series.iloc[:, 0].dropna().tolist()
    detect_time = t_timestamps[detect_idx-1] if detect_idx > 0 else None
    
    time_format = r"^\d+:\d+\.\d+$"
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if re.match(time_format, detect_time):
        dt_hr, dt_mins = detect_time.split(':')
        dt_mins = float(dt_mins)
        dt_mint = int(dt_mins)
        dt_secs = int(round((dt_mins - dt_mint) * 60))
        if dt_secs == 60:
            dt_secs = 0
            dt_mins += 1
        detect_time = f"{today_date} {dt_hr}:{int(dt_mins):02d}:{int(dt_secs):02d}"
    print("Detected at:", detect_time)
    y = p1.to_numpy().reshape(-1, 1)
    
    if 0 < event_start_idx <= len(data):
        start_idx = max(event_start_idx - 10, 0)
        extracted_data = data[start_idx:]
        extracted_time = t[start_idx:]
        print("\nExtracted Data:")
        print(f" Start time: {detection_time:.2f}s")
        print(f" End time: {t[-1]:.2f}s")
        print(f" Duration: {t[-1] - detection_time:.2f}s\n")
    else:
        print("\nNo valid detection - nothing to extract\n")
    
    if 'extracted_data' in locals() and len(extracted_data) > 0:
        m = 1 #initialy m=4
        p = 1 #p=2
        threshold = 0.5
        x_osc = extracted_data - mu_ambient
        x = np.arange(len(x_osc)).reshape(-1, 1)
        zero_crossings = []
        for i in range((len(x_osc) - 1)):
            xosc0 = x_osc[i]
            xosc1 = x_osc[i+1]
            if xosc0 < 0 and xosc1 > 0:
                zero_crossings.append(i)
            elif xosc0 > 0 and xosc1 < 0:
                zero_crossings.append(i)
            elif xosc0 == 0 and xosc1 > xosc0:
                zero_crossings.append(i)
            elif xosc0 == 0 and xosc1 < xosc0:
                zero_crossings.append(i)
        if len(zero_crossings) >= m + p:
            num_windows = len(zero_crossings) - (m + p - 1)
            DCR = np.zeros(num_windows)
            flag = np.zeros(num_windows)
            consecutive_count = 0
            for n in range(m + p-1, len(zero_crossings)):
                current_idx = n - (m + p - 1)
                current_start = zero_crossings[n - m + 1]
                current_end = zero_crossings[n]
                preceding_start = zero_crossings[n - m - p + 1]
                preceding_end = zero_crossings[n - p]
                preceding_mean = np.mean(extracted_data[preceding_start:preceding_end+1])
                current_window = extracted_data[current_start:current_end+1]
                x_osc_window = current_window - preceding_mean
                A_plus = np.sum(x_osc_window[x_osc_window > 0])
                A_minus = abs(np.sum(x_osc_window[x_osc_window < 0]))
                if max(A_plus, A_minus) > 0:
                    DCR[current_idx] = min(A_plus, A_minus) / max(A_plus, A_minus)
                if DCR[current_idx] > threshold:
                    consecutive_count += 1
                    if consecutive_count >= 3:
                        flag[current_idx] = 1
                else:
                    consecutive_count = 0
                    continue
            print("\nDCR Statistics:")
            print(f" Average: {np.mean(DCR):.2f}")
            print(f" Minimum: {np.min(DCR):.2f}")
            print(f" Maximum: {np.max(DCR):.2f}\n")
        else:
            print("\nInsufficient zero crossings for DCR calculation\n")
            print(f"No oscillation detected. Running again.")
    
    #%%
    def load_data(file_name, prefix):
        print(f"Loading {prefix.upper()}....")
        df = pd.read_excel(file_name, skiprows=0, nrows=30000)
        df = df.iloc[:, 1:]
        bus_ids = df.columns.tolist()
        df = df.transpose().interpolate(method="linear", axis=1)
        data_list = [df.iloc[[i]] for i in range(len(bus_ids))]
        for i, data in enumerate(data_list, start=1):
            var_name = f"{prefix}{i}"
            globals()[var_name] = data
        print(f"{prefix.upper()}: {len(data_list)} variables loaded.")
        return data_list, bus_ids
    
    def pmu_dir(data_list, bus_ids, pmu_file):
        try:
            pmu_dir = pd.read_excel(pmu_file, usecols='B', skiprows=0, header=None).squeeze()
            pmu_dir_dict = {str(bus)[:23]: dir_val for bus, dir_val in pmu_dir.items()}
            adjusted_data = data_list.copy()
            for i, bus_id in enumerate(bus_ids):
                truncated_id = str(bus_id)[:23]
                direction = pmu_dir_dict.get(truncated_id, 1)
                if direction == -1:
                    adjusted_data[i] *= -1
            return adjusted_data
        except FileNotFoundError:
            print(f"PMU direction data not found. Skipping adjustment.")
            return data_list.copy()
    
    P_data, P_bus_ids = load_data("Data_Raw_P.xlsx", "p")
    Q_data, Q_bus_ids = load_data("Data_Raw_Q.xlsx", "q")
    V_all, V_bus_ids = load_data("Data_Raw_V.xlsx", "v")
    A_all, A_bus_ids = load_data("Data_Raw_A.xlsx", "a")
    
    # print("Applying PMU direction to P & Q....")
    # P_all = pmu_dir(P_data, P_bus_ids, "Bus_names.xlsx")
    # Q_data = pmu_dir(Q_data, Q_bus_ids, "Bus_names.xlsx")
    ma = P_all + Q_data + V_all + A_all
    ma_array = np.array(ma)
    print("ma_array.shape", (ma_array.shape))
    
    N_bus = (ma_array.shape[0] / 4)
    null_matrix = np.zeros_like(ma_array[0])
    final_matrix = np.zeros((ma_array.shape[0], 1), dtype=np.complex128)
    
    ind = 0
    for u in range(ma_array.shape[0]):
        t = np.arange(0, round((ma_array[u].shape[1] - 1) * 0.04, 2) + 0.04, 0.04)
        if (ma_array[u].all() == null_matrix.all()):
            continue
        y = ma_array[u]
        yreshape = y.reshape(-1, 1)
        print(y)
        print("Loc..", u + 1)
    
        Nt = t[::2]
        dt = Nt[1] - Nt[0]
        Ny = yreshape[::2]
        Ny = Ny.flatten()
        N = len(Ny)
        L = int(np.floor(N / 2) - 1)
        Y = np.zeros((N - L, L + 1), dtype=float)
        for i in range(N - L):
            for j in range(L + 1):
                Y[i][j] = Ny[i + j]
    
        U, S, Vt = la.svd(Y, full_matrices=True)
        S_matrix = np.diag(S)
        N_SV = 10 * np.log10((S) / (S[0]))
    
        sv = None
        for i in range(len(N_SV) - 1, -1, -1):
            if N_SV[i] > -35.5:
                sv = i + 1
                break
        if sv is None:
            sv = 0
    
        M = sv
        print("The value of M is", M)
        UU = U[:, :M]
        print(UU.shape)
        VV = Vt[:M, :].T
        print(VV.shape)
        SS = np.diag(S[:M])
        print(SS.shape)
        VV_trans = np.conjugate(VV.T)
        YY = UU @ SS @ VV_trans
    
        V2 = VV[:-1, :]
        V1 = VV[1:, :]
        Y1 = UU @ SS @ np.conjugate(V1.T)
        Y2 = UU @ SS @ np.conjugate(V2.T)
    
        A, B = la.eig(np.linalg.pinv(V2) @ V1)
        A = np.diag(A)
        Z0 = A[:M, :M]
        zi = np.diag(Z0)
        zi1 = np.log(zi)
        P = zi1 / dt
    
        ZII = np.zeros((N, M), dtype=np.complex128)
        R_list = []
        for k in range(1, N + 1):
            ZII[k - 1, :] = zi ** (k - 1)
        R = np.linalg.pinv(ZII) @ Ny.T
        R_list.append(R)
    
        # yr = np.zeros_like(t, dtype=np.complex128)
        # for k in range(M):
        #     yr += R[k] * np.exp(P[k] * t)
        yr = np.zeros_like(t, dtype=np.complex128)
        for k in range(M):
            yr += R[k] * np.exp(P[k] * t)
        
        # Store reconstructed signal (real part only, MW)
        globals()[f"yr_bus_{u+1}"] = np.real(yr)
    
        plt.plot(yr)
    
        agg = []
        for i in range(len(P)):
            # if 0.2 < np.imag(P[i]) / (2 * np.pi) < 5:
            if 0.2 < np.imag(P[i]) / (2 * np.pi) < 5:
                agg.append(P[i])
    
        mode_meter = np.imag(P) / (2 * np.pi)
        mode_meter_1 = []
        for i in range(len(mode_meter)):
            # if 0.2 < np.imag(P[i]) / (2 * np.pi) < 2:
            if 0.2 < np.imag(P[i]) / (2 * np.pi) < 5:
                mode_meter_1.append(P[i])
    
        if len(agg) == 0:
            nothing = 'nothing'
        else:
            a = np.real(agg[0])
            b = agg[0]
            for j in range(len(agg)):
                if np.real(agg[j]) < a:
                    a = np.real(agg[j])
                    b = agg[j]
            print(b)
            final_matrix[ind] = b
            ind = ind + 1
    
    #     # End of conditional block for agg
    ##                print(final_matrix.shape)
    non_zero_rows = np.any(final_matrix != 0, axis=1)
    non_zero_cols = np.any(final_matrix != 0, axis=0)
    final_matrix = final_matrix[non_zero_rows][:, non_zero_cols]
    ##                print(final_matrix.shape)
    
    S = final_matrix
    S_real = np.real(S)
    S_imag = np.imag(S)
    ##                print(S_real.shape)
    ##                print(S_imag.shape)
    
    # Initialize an empty list to store the results (equivalent to I = [])
    I = []
    
    # Iterate over each element in the flattened array of S
    for s in S.flatten():
        # Append the rounded imaginary part to I
        I.append(round(np.imag(s), 2))
    
    # Convert the list I to a numpy array, reshape it according to the dimensions you need
    I = np.array(I).reshape(S.shape)
    
    print("Shape of I = ", I.shape)
    ##                print(I)
    
    Md, F = stats.mode(I)
    Md = Md[0]
    F = F[0]
    print("Md", Md)
    print("F", F)
    
    dom_mode = 0
    dom_mode1 =0
    
    if F >= 2:
        dom_mode = round((Md/(2*np.pi)),2)
        dom_mode1 = round(Md,2)
    
    
    final_matrix_1 = []
    # print(trimmed_final_matrix[93].item())
    
    for i in range(final_matrix.shape[0]):
      z = final_matrix[i].item()
      z_real_round = round(z.real,2)
      z_imag_round = round(z.imag,2)
      z_round = complex(z_real_round,z_imag_round)
      final_matrix_1.append(z_round)
    
    print("final_matrix_1", final_matrix_1)
    print("length of final_matrix_1", len(final_matrix_1))
    
    
    
    J = []
    for i in range(final_matrix.shape[0]):
     z = final_matrix_1[i]
     if(z.imag==dom_mode1):
       J.append(final_matrix_1[i])
    print(J)
    
    real_J = np.real(J)
    print(real_J)
    if(len(real_J)!=0):
        min_damp = min(real_J)
        print("min_damp = ", min_damp)
        
        
    
    
    
    r_r = 25  # reporting rate of emulated PMUs per second
    T_window = 2  # window size in seconds
    S_window = T_window * r_r  # sample length of DEF window
    
    
    ##                Base = pd.read_excel("Data_Raw_Base.xlsx", skiprows=0, nrows=4500)
    ##                P_base = Base["pbase"]
    ##                V_base = Base["vbase"]
    ##
    ##                def per_unit(prefix, base_values):
    ##                    for i in range(len(base_values)):  
    ##                        var_name = f"{prefix}{i+1}"
    ##                        if var_name in globals():
    ##                            globals()[var_name] = globals()[var_name] / base_values[i]
    ##                        else:
    ##                            print(f"Warning: {var_name} not found in globals.")
    ##
    ##                per_unit("p", P_base)
    ##                per_unit("v", V_base)
    ##                per_unit("q", P_base)
    ##
    ##                V_bus = [globals()[f"v{i+1}"] for i in range(len(V_base)) if f"v{i+1}" in globals()]
    ##                P_bus = [globals()[f"p{i+1}"] for i in range(len(P_base)) if f"p{i+1}" in globals()]
    ##                Q_bus = [globals()[f"q{i+1}"] for i in range(len(P_base)) if f"q{i+1}" in globals()]
    ##                theta_bus = [globals()[f"a{i+1}"] for i in range(len(V_base)) if f"a{i+1}" in globals()]
    
    def V_rated_P_pu(df: pd.DataFrame) -> tuple:
        V_rated = []
        P_map = {
            400: 874,
            220: 214,
            132: 84
        }
        P_pu = []
    
        for col in df.columns[1:]:
            match = re.search(r'(\d{3})', col)
            if match:
                voltage = int(match.group(1))
                V_rated.append(voltage)
                P_pu.append(P_map.get(voltage, None))
        return V_rated, P_pu
    Vdata = pd.read_excel("Data_Raw_V.xlsx")
    Vrated, Prated_pu = V_rated_P_pu(Vdata)
    Vrated_pu = np.round(np.array(Vrated) / np.sqrt(3), 3)
    P_bus = P_all.copy()
    Q_bus = Q_data.copy()
    V_bus = V_all.copy()
    theta_bus = A_all.copy()
    
    print("Per unitizing P, Q & V....")
    for i in range(len(P_all)):
        P_bus[i] = P_all[i] / 100
        Q_bus[i] = Q_data[i] / 100
        V_bus[i] = V_all[i] / 100
        globals()[f"p{i+1}"] = P_bus[i]
        globals()[f"q{i+1}"] = Q_bus[i]
        globals()[f"v{i+1}"] = V_bus[i]
    print("Per unitization complete.")
    
    ll = 1
    fr = dom_mode
    print("fr", fr)
    Nr = r_r / 2
    print("Nr", Nr)
    
    # V_bus = np.vstack([df.values for df in V_bus])
    # theta_bus = np.vstack(df.values for df in theta_bus)
    V_bus = np.array(V_bus)
    theta_bus = np.array(theta_bus)
    # FIR filter design
    low_cutoff = 0.7*(fr / Nr)
    high_cutoff = 1.3 * (fr / Nr)
    #window = (signal.windows.chebwin(41, at=30))
    window = ('chebwin', 30)
    b = signal.firwin(41, [low_cutoff, high_cutoff], window = window, pass_zero= False)
    del_V = np.zeros_like(V_bus)
    del_theta = np.zeros_like(theta_bus)
    
    for i in range (0, int(N_bus)):
        del_V[i, :] = signal.filtfilt(b, [1, 0], V_bus[i, :])
        del_theta[i, :] = signal.filtfilt(b, [1, 0], theta_bus[i, :])
    
    P_bus = np.array(P_bus)
    Q_bus = np.array(Q_bus)
    del_P = np.zeros_like(P_bus)
    del_Q = np.zeros_like(Q_bus)
    
    for i in range (0, int(N_bus)):
        del_P[i, :] = signal.filtfilt(b, [1, 0], P_bus[i, :])
        del_Q[i, :] = signal.filtfilt(b, [1, 0], Q_bus[i, :])
    del_P = np.squeeze(del_P, axis=1)
    del_Q = np.squeeze(del_Q, axis=1)
    
    V3 = np.zeros_like(del_V)
    Theta3 = np.zeros_like(del_theta)
    for i in range (0, int(N_bus)):
        V3[i, :] = signal.detrend(del_V[i, :])
        Theta3[i, :] = signal.detrend(del_theta[i, :])
    
    del_V = []
    del_theta = []
    
    for i in range (0, int(N_bus)):
        del_V.append(V3[i, :])
        del_theta.append(Theta3[i, :])
    del_V = np.squeeze(del_V, axis=1)
    del_theta = np.squeeze(del_theta, axis=1)
    
    P3 = np.zeros_like(del_P)
    Q3 = np.zeros_like(del_Q)
    
    for i in range (0, int(N_bus)):
        P3[i, :] = signal.detrend(del_P[i, :])
        Q3[i, :] = signal.detrend(del_Q[i, :])
    
    del_P = []
    del_Q = []
    
    for i in range (0, int(N_bus)):
        del_P.append(P3[i, :])
        del_Q.append(Q3[i, :])
    
    T_window = 2
    S_window = T_window * r_r
    N = del_V[0].size
    N_t = np.ceil((N - S_window - 1) / S_window)
    V_bus2D = np.squeeze(V_bus, axis=1)
    n = 0
    V_T = np.zeros((int(N_bus), int(N_t)))
    
    
    #........................................................................
    ##Finding index corresponding to dominant mode.
    
    P_imag = np.round(np.imag(P),2)
    # J_value = np.imag(J[0])
    J_value = 2.834e+01
    P_index = np.where(P_imag == J_value)
    A_corr_P = R[P_index]
    
    #...........................................................................
    
    
    
    n = 0
    V_T = np.zeros((int(N_bus), int(N_t)))
    print(V_T.shape)
    V_bus =np.squeeze(V_bus)
    print(V_bus.shape)
    # Loop over the starting index of each window
    for ii in range(1, N - (S_window), (S_window)):
        n += 1
        # Loop over the sample indices within the current window
        for jj in range((ii), (ii + S_window)):
            for k in range(int(N_bus)):
                V_T[k, n - 1] += V_bus2D[k, jj]  # accumulate the voltage values
    
    V_avg = V_T / S_window
    
    print(V_avg)
    print(V_avg.shape)
    
    
    DEFS1 = np.zeros((int(N_bus),int(N_t)),dtype=float)
    DEFS2 = np.zeros((int(N_bus),int(N_t)),dtype=float)
    DEFST = np.zeros((int(N_bus),int(N_t)),dtype=float)
    
    DEF1=np.zeros((int(N_bus),1),dtype=float)
    DEF2=np.zeros((int(N_bus),1),dtype=float)
    DEFT=np.zeros((int(N_bus),1),dtype=float)
    
    DER = np.zeros((1,int(N_bus)),dtype=float)
    
    n=0
    for ii in range(1, N - (S_window), (S_window)):
        n += 1
        # Loop over the sample indices within the current window
        for jj in range((ii), (ii + S_window)):
            for kk in range(int(N_bus)):
              t_b=kk+1
              # DEF1[kk]=DEF1[kk]+  del_P[kk][jj]*(del_theta[t_b-1][jj+1]-del_theta[t_b-1][jj-1])*((np.pi)/180)/2
              # DEF2[kk]=DEF2[kk]+  del_Q[kk][jj]*(del_V[t_b-1][jj+1]-del_V[t_b-1][jj-1])/(2*V_avg[t_b-1][n-1])
              # DER[0][kk] = DEF1[kk] + DEF2[kk]
              
              DEF1[kk]=DEF1[kk]+ 0.995037 * del_P[kk][jj]*(del_theta[t_b-1][jj+1]-del_theta[t_b-1][jj-1]) + 0.0995 * del_P[kk][jj] * (del_V[t_b-1][jj+1]-del_V[t_b-1][jj-1])/(V_avg[t_b-1][n-1])
              DEF2[kk]=DEF2[kk]+ 0.995037 * del_Q[kk][jj]*(del_V[t_b-1][jj+1]-del_V[t_b-1][jj-1])/(V_avg[t_b-1][n-1]) - 0.0995 * del_Q[kk][jj] * (del_theta[t_b-1][jj+1]-del_theta[t_b-1][jj-1])
              DER[0][kk] = DEF1[kk] + DEF2[kk]
    
        DEFS1[:,n-1] = DEF1[:,0]
        DEFS2[:,n-1] = DEF2[:,0]
        DEFST[:,n-1] = DEF1[:,0] + DEF2[:,0]
    
    
    DEF1_bus = np.zeros((int(N_bus),int(N_t)),dtype=float)
    DEF2_bus = np.zeros((int(N_bus),int(N_t)),dtype=float)
    DEFT_bus = np.zeros((int(N_bus),int(N_t)),dtype=float)
    
    for ii in range(0,int(N_bus)):
      DEF1_bus[ii,:] = DEFS1[ii,:]
      DEF2_bus[ii,:] = DEFS2[ii,:]
      DEFT_bus[ii,:] = DEFST[ii,:]
    
    N_gen = N_bus
    DEF1_gen = np.zeros((int(N_gen),int(N_t)),dtype=float)
    DEF2_gen = np.zeros((int(N_gen),int(N_t)),dtype=float)
    DEFT_gen = np.zeros((int(N_gen),int(N_t)),dtype=float)
    
    for ii in range(0,int(N_gen)):
      DEF1_gen[ii,:]= DEF1_bus[ii,:]
      DEF2_gen[ii,:]= DEF2_bus[ii,:]
      DEFT_gen[ii,:]= DEFT_bus[ii,:]
    
    T_time = t[-1]
    t1 = np.arange(T_window, T_window * N_t + T_window, T_window)
    t_bar = t1[0]
    t2 = t1 - t_bar
    
    pp = np.zeros((int(N_gen),2),dtype=float)
    DEF_pp_lin_fit = np.zeros((int(N_gen),len(t1)),dtype=float)
    
    for ii in range(0,int(N_gen)):
      if np.isnan(DEFT_gen[ii,:]).all():
          pp[ii,:]=np.nan
          continue
      coefficients = np.polyfit(t2, DEFT_gen[ii, :], 1)
      pp[ii, :] = coefficients
    
      DEF_pp_lin_fit[ii, :] = coefficients[0] * (t1 - t_bar) + coefficients[1]
      
    
    # plt.figure()
    # plt.title("Figure - 1")
    # plt.plot(t1, DEF_pp_lin_fit[ii, :], label=f'Generator 1')
    # for ii in range(1, int(N_gen)):
    #     plt.plot(t1, DEF_pp_lin_fit[ii, :], label=f'Generator {ii + 1}')
    
    # plt.ylabel('DEF1 after linear LS fitting: all generators-1')
    # plt.xlabel('time in seconds')
    # plt.show()
    
    pq = np.zeros((int(N_gen),2),dtype=float)
    DEF_pq_lin_fit = np.zeros((int(N_gen),len(t1)),dtype=float)
    
    for ii in range(0,int(N_gen)):
      if np.isnan(DEFT_gen[ii,:]).all():
          pq[ii,:]=np.nan
          continue
      coefficients2 = np.polyfit(t2, DEFT_gen[ii, :], 1)
      pq[ii, :] = coefficients2
    
      DEF_pq_lin_fit[ii, :] = coefficients2[0] * (t1 - t_bar) + coefficients2[1]
      
    
    # plt.figure()
    # plt.title("Figure - 2")
    # plt.plot(t1, DEF_pq_lin_fit[0, :], label=f'Generator 1')
    # for ii in range(1, int(N_gen)):
    #     plt.plot(t1, DEF_pq_lin_fit[ii, :], label=f'Generator {ii + 1}')
    
    # plt.ylabel('DEF1 after linear LS fitting: all generators-2')
    # plt.xlabel('time in seconds')
    # plt.show()
    
    slope_pp = np.zeros((int(N_gen),1),dtype=float)
    slope_pq = np.zeros((int(N_gen),1),dtype=float)
    for ii in range(0,int(N_gen)):
      slope_pp[ii][ll-1] = pp[ii][0]
      slope_pq[ii][ll-1] = pp[ii][0]
    
    mm = DEFST[1][n-1]
    mm_index = 1;
    for i in range(0,int(N_bus)):
      if(DEFST[i][n-1]>mm):
        mm = DEFST[i][n-1]
        mm_index = i
    
    
    if DEFS1[mm_index][n-1] > DEFS2[mm_index][n-1]:
        suspect_status = "Governor" 
        print("Suspect - Governor") # Store the text for Governor
        suspect = 1
    elif DEFS1[mm_index][n-1] < DEFS2[mm_index][n-1]:
        suspect_status = "Exciter"    # Store the text for Exciter
        print("Suspect - Exciter")
        suspect = 2
    else:
        suspect_status = "Both Gov and Exc"  # Store the text for both
        print("Suspect - Both Gov and Exc")
        suspect = 3
    
    der_slope = DEFST[:,n-1]
    der_slope = der_slope.flatten()
    der_slope_par = np.partition(der_slope,3)
    nan_indices = np.where(np.isnan(der_slope_par))[0]
    der_slope_par = np.delete(der_slope_par,nan_indices)
    der_slope_par = np.sort(der_slope_par)
    max_der = der_slope_par[-3:]
    min_der = der_slope_par[:3]
    
    max_index = []
    min_index = []
        
    for i in range(len(max_der)):
      index = np.where(DEFST[:,n-1] == max_der[i])[0]
      max_index.append(index[0])
    
    for i in range(len(min_der)):
      index2 = np.where(DEFST[:,n-1]==min_der[i])[0]
      min_index.append(index2[0])
      
    
    max_matrix = []
    
    for i in range(len(max_index)):
      max_matrix.append(final_matrix[max_index[i]])
    
    min_matrix = []
    
    for i in range(len(min_index)):
      min_matrix.append(final_matrix[min_index[i]])
    
    DEFST_combine = []
    for i in range(len(DEFST[:,n-1])):
      DEFST_combine.append(t1*DEFST[i][n-1])
      
      
    
    
    z = [final_matrix, suspect, dom_mode, min_damp, max_matrix, min_matrix, t1, DEFST_combine, t2, DEFS1[mm_index,:], DEFS2[mm_index,:]]
    z[1] = int(z[1])                      # Convert to Python int
    z[2] = float(z[2])                   # Convert to Python float
    z[3] = float(z[3])
    final_matrix_real_part = np.real(z[0])  # Real part of complex128
    final_matrix_imag_part = np.imag(z[0])  # Imaginary part of complex128
    # Convert to native Python types
    # Convert each element in the array to a float using astype
    final_matrix_real_part = final_matrix_real_part.astype(float).tolist()
    # Convert each element in the array to a float using astype
    final_matrix_imag_part = final_matrix_imag_part.astype(float).tolist()
    
    max_mat = np.concatenate(z[4]).ravel()     #for max matrix
    max_matrix_real = np.real(max_mat).tolist()
    max_matrix_imag = np.imag(max_mat).tolist()
    
    min_mat = np.concatenate(z[5]).ravel()     #for min matrix
    min_matrix_real = np.real(min_mat).tolist()
    min_matrix_imag = np.imag(min_mat).tolist()
    
    t1_table = (z[6]).tolist()  #for t1
    
    t2_table = (z[8]).tolist()  #for t2
    
    DEFS1_list = (z[9]).tolist() #for DEFS1[mm_index, :]
    
    DEFS2_list = (z[10]).tolist()   #for DEFS2[mm_index, :]
    
    der_slope = der_slope.astype(float).tolist()
    
    DEFST_column_count = len(z[7])
    columns = ', '.join([f"column_{i+1} FLOAT" for i in range(DEFST_column_count)])
    
    mode_p = np.imag(P)/(2*np.pi)
    mode_meter_0_2 = mode_p[(mode_p >= 0.2) & (mode_p < 2)].tolist()
    mode_meter_2_4 = mode_p[(mode_p >= 2) & (mode_p < 4)].tolist()
    mode_meter_4_6 = mode_p[(mode_p >= 4) & (mode_p < 6)].tolist()
    mode_meter_6_12 = mode_p[(mode_p >= 6) & (mode_p < 12)].tolist()
    
    #%%
    conn_params = {
        'user': 'postgres',
        'password': 'password123',
        'host': 'localhost',
        'port': '5432',
        'database': 'Osi_Tool_DB'
    }
    
    # Connect to the PostgreSQL database
    # connection = psycopg2.connect(**conn_params)
    # cursor = connection.cursor()
    
    def create_database_if_not_exists(db_name, user, password, host, port):
        """Check if a PostgreSQL database exists, and create it if it does not."""
        try:
            # Connect to the default 'postgres' database
            conn = psycopg2.connect(dbname="postgres", user=user, password=password, host=host, port=port)
            conn.autocommit = True  # Enable autocommit for DDL operations
            cursor = conn.cursor()
    
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            exists = cursor.fetchone()
    
            if not exists:
                # Create the database
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
                print(f"Database '{db_name}' created successfully.")
            else:
                print(f"Database '{db_name}' already exists.")
    
            cursor.close()
            conn.close()
        
        except psycopg2.Error as e:
            print(f"Error: {e}")
    
    # Ensure the database exists before connecting
    create_database_if_not_exists(conn_params['database'], conn_params['user'], conn_params['password'], conn_params['host'], conn_params['port'])
    
    # Now connect to the actual database
    try:
        connection = psycopg2.connect(**conn_params)
        cursor = connection.cursor()
        print(f"Connected to database '{conn_params['database']}' successfully.")
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
    
    
    # Connect to the PostgreSQL database
    connection = psycopg2.connect(**conn_params)
    cursor = connection.cursor()
    
    # Create table if it doesn't exist
    create_table_query = """ 
        DROP TABLE IF EXISTS Suspect_table;
        CREATE TABLE Suspect_table (
        id SERIAL PRIMARY KEY,
        suspect INTEGER,                   -- Changed to INTEGER for int
        dom_mode FLOAT8,                  -- FLOAT8 for float64
        min_damp FLOAT8                    -- FLOAT8 for float64
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    
    create_table_query = """ 
        DROP TABLE IF EXISTS final_matrix;
        CREATE TABLE final_matrix (
        id SERIAL PRIMARY KEY,
        final_matrix_real_part FLOAT8,                   -- FLOAT8 for float64
        final_matrix_imag_part FLOAT8                  -- FLOAT8 for float64
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    
    create_table_query = """ 
        DROP TABLE IF EXISTS max_matrix;
        CREATE TABLE max_matrix (
        id SERIAL PRIMARY KEY,
        max_matrix_real FLOAT8,                   -- FLOAT8 for float64
        max_matrix_imag FLOAT8                  -- FLOAT8 for float64
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    
    create_table_query = """
        DROP TABLE IF EXISTS min_matrix;
        CREATE TABLE min_matrix (
        id SERIAL PRIMARY KEY,
        min_matrix_real FLOAT8,                   -- FLOAT8 for float64
        min_matrix_imag FLOAT8                  -- FLOAT8 for float64
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    
    create_table_query = """
        DROP TABLE IF EXISTS t1;
        CREATE TABLE t1 (
        id SERIAL PRIMARY KEY,
        t1_table FLOAT8                  -- FLOAT8 for float64
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    create_table_query = """
        DROP TABLE IF EXISTS t2;
        CREATE TABLE t2 (
        id SERIAL PRIMARY KEY,
        t2_table FLOAT8                  -- FLOAT8 for float64
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    create_table_query = """
        DROP TABLE IF EXISTS DEFS1;
        CREATE TABLE DEFS1 (
        id SERIAL PRIMARY KEY,
        DEFS1_list FLOAT8                  -- FLOAT8 for float64
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    create_table_query = """
        DROP TABLE IF EXISTS DEFS2;
        CREATE TABLE DEFS2 (
        id SERIAL PRIMARY KEY,
        DEFS2_list FLOAT8                  -- FLOAT8 for float64
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    # Generate column definitions for DEFST_combine
    num_columns = len(DEFST_combine)
    bus_names_df = pd.read_excel('Data_Raw_P.xlsx', sheet_name='Sheet1')
    bus_names = bus_names_df.columns[1:].tolist()
    bus_names = [b.replace (".","_") for b in bus_names]
    columns = ",\n        ".join([f"{bus_names[i]} FLOAT8" for i in range(num_columns)])
    
    # Final table creation query
    create_table_query = f"""
        DROP TABLE IF EXISTS DEFST_combined;
        CREATE TABLE DEFST_combined (
            row_id SERIAL PRIMARY KEY,
            {columns},
            t1 FLOAT8,
            time_series TIMESTAMP
        );
    """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    
    
    # SQL statement to create the suspect_status table
    create_table_query = """
    DROP TABLE IF EXISTS suspect_status;
    CREATE TABLE suspect_status (
        id SERIAL PRIMARY KEY,
        time_series TIMESTAMP,  -- Assuming you want to associate this with a time series
        suspect TEXT
    );
    """
    
    # Execute the SQL command to create the table
    cursor.execute(create_table_query)
    
    # Commit the changes to the database
    connection.commit()
    
    
    # Create the high_der_slope table with separate columns for positive and negative slopes
    create_table_query = """
    DROP TABLE IF EXISTS high_der_slope;
    CREATE TABLE high_der_slope (
        id SERIAL PRIMARY KEY,
        time_series TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Auto timestamp
        positive_slope FLOAT8,  -- Stores positive slopes
        negative_slope FLOAT8   -- Stores negative slopes
    );
    """
    
    # Execute the table creation query
    cursor.execute(create_table_query)
    connection.commit()
    
    
    
    ## bus number
    
    # SQL statement to create the high_der_slope table
    # SQL statement to create the high_der_slope table
    create_table_query = """
    DROP TABLE IF EXISTS bus_number;
    CREATE TABLE bus_number (
        id SERIAL PRIMARY KEY,
        time_series TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        bus_number  INT,
        positive_slope FLOAT8
        
    );
    """
    
    # Execute the table creation query
    cursor.execute(create_table_query)
    connection.commit()
    
    
    
    # create_table_query = """
    # CREATE TABLE IF NOT EXISTS oscillation_data (
    #     id SERIAL PRIMARY KEY,
    #     time_series TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    #     bus_number INT,
    #     osci_P FLOAT8
    # );
    # """
    # cursor.execute(create_table_query)
    # connection.commit()
     ##### mode metrer
    create_table_query = """
    DROP TABLE IF EXISTS mode_meter_data;
    CREATE TABLE mode_meter_data (
        id SERIAL PRIMARY KEY,
        range_0_2 FLOAT8,      -- Column for mode_meter_0_2 values
        range_2_4 FLOAT8,      -- Column for mode_meter_2_4 values
        range_4_6 FLOAT8,      -- Column for mode_meter_4_6 values
        range_6_12 FLOAT8      -- Column for mode_meter_6_12 values
    );
    """
    
    # Execute the table creation query
    cursor.execute(create_table_query)
    connection.commit()
    
    
    print("Table created or already exists.")
    
    
    #%%
    connection = psycopg2.connect(**conn_params)
    cursor = connection.cursor()
    # Define the insert query
    # Extract values from the z array
    suspect = z[1]                     # Integer
    dom_mode = z[2]                    # Float64
    min_damp = z[3]
    final_matrix_real_part
    final_matrix_imag_part                   # Float64
    # Insert data into the table
    insert_query = """INSERT INTO Suspect_table (suspect, dom_mode, min_damp) 
                          VALUES (%s, %s, %s);"""
    
    cursor.execute(insert_query, (suspect, dom_mode, min_damp))
    connection.commit()
    
      
    #detection timestamp
    #detect_time = "17-05-2025  14:40:00" #placeholder
    create_table_query = """ 
        DROP TABLE IF EXISTS detection_timestamp;
        CREATE TABLE detection_timestamp(
        id SERIAL PRIMARY KEY,
        detection_time TEXT NOT NULL                 
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    insert_query ="""INSERT INTO detection_timestamp (detection_time) 
                     VALUES (%s)"""
    cursor.execute(insert_query, (detect_time,))
    connection.commit()
    #%%    
    #for final matrix
    flattened_real = [float(item[0]) for item in final_matrix_real_part]
    flattened_imag = [float(item[0]) for item in final_matrix_imag_part]
    
    for real, imag in zip(flattened_real, flattened_imag):
        insert_query = """INSERT INTO final_matrix(final_matrix_real_part, final_matrix_imag_part)
                          VALUES (%s, %s);"""
        cursor.execute(insert_query, (real, imag))
        
    connection.commit()
    
    #for max matrix
    mat_max_real = [float(item) for item in max_matrix_real]
    mat_max_imag = [float(item) for item in max_matrix_imag]
    
    for real, imag in zip(mat_max_real, mat_max_imag):
        insert_query = """INSERT INTO max_matrix(max_matrix_real, max_matrix_imag)
                          VALUES (%s, %s);"""
        cursor.execute(insert_query, (real, imag))
        
    connection.commit()
    
    #for min matrix
    mat_min_real = [float(item) for item in min_matrix_real]
    mat_min_imag = [float(item) for item in min_matrix_imag]
    
    for real, imag in zip(mat_min_real, mat_min_imag):
        insert_query = """INSERT INTO min_matrix(min_matrix_real, min_matrix_imag)
                          VALUES (%s, %s);"""
        cursor.execute(insert_query, (real, imag))
        
    connection.commit()
    
    #for t1
    for values in zip(t1_table):
        insert_query = """INSERT INTO t1 (t1_table) 
                          VALUES (%s);"""
        cursor.execute(insert_query, (values))
    
    connection.commit()
    
    #for t2
    for values in zip(t2_table):
        insert_query = """INSERT INTO t2 (t2_table) 
                          VALUES (%s);"""
        cursor.execute(insert_query, (values))
    
    connection.commit()
    
    #for DEFS1
    for values in zip(DEFS1_list):
        insert_query = """INSERT INTO DEFS1 (DEFS1_list) 
                          VALUES (%s);"""
        cursor.execute(insert_query, (values))
    
    connection.commit()
    
    #for DEFS2
    for values in zip(DEFS2_list):
        insert_query = """INSERT INTO DEFS2 (DEFS2_list) 
                          VALUES (%s);"""
        cursor.execute(insert_query, (values))
    
    connection.commit()
    #%%
    # Load timestamps from Excel
    
    
    timestamps_df = pd.read_excel('Data_Raw_P.xlsx', usecols='A', skiprows=0)
    timestamps = timestamps_df.iloc[:, 0].dropna().tolist()
    #%%
    if re.match(time_format, timestamps[0]):
        print("Timestamps detected to be of HH:MM.M")
        print("Converting to YYYY-MM-DD HH:MM:SS")
        for i in range(len(timestamps)):
            ti = timestamps[i]
            hr, mins = ti.split(':')
            mins = float(mins)
            mint = int(mins)
            secs = int(round((mins - mint) * 60))
            if secs == 60:
                secs = 0
                mins += 1
            ti_new = f"{today_date} {hr}:{int(mins):02d}:{int(secs):02d}"
            timestamps[i] = ti_new
            
        print("Timestamps converted.")
    else:
        print("Timestamps already in YYYY-MM-DD format.")
        
    
     #%%   
    defst_time = []
    defst_range = len(DEFST_combine[0])
    n_time = len(timestamps)
    n_defst = len(DEFST_combine[0])
    defst_range = int(n_time / n_defst)
    for i in range (0, (n_time - defst_range), defst_range):
        t = timestamps[i]
        defst_time.append(t)
    
    
    # Number of columns in DEFST_combine
    num_columns = len(DEFST_combine)
    
    
    # Insert rows
    
    for i in range(len(DEFST_combine[0])):
        row_values = [float(DEFST_combine[j][i]) for j in range(num_columns)]
        row_values.append(float(t1[i]))
        row_values.append(defst_time[i])
    
        column_names = [f'{bus_names[j]}' for j in range(num_columns)] + ['t1', 'time_series']
        placeholders = ', '.join(['%s'] * len(row_values))
        insert_query = f"""
            INSERT INTO DEFST_combined ({', '.join(column_names)})
            VALUES ({placeholders});
        """
        cursor.execute(insert_query, row_values)
    
    connection.commit()
    
    
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare the insert query with ON CONFLICT to overwrite the data
    insert_query = """
        INSERT INTO suspect_status (id, time_series, suspect) 
        VALUES (1, %s, %s)
        ON CONFLICT (id)
        DO UPDATE SET suspect = EXCLUDED.suspect, time_series = EXCLUDED.time_series;
    """
    cursor.execute(insert_query, (current_time, suspect_status))
    connection.commit()
    
    
    
    
    # Load the bus names from the Excel sheet
    # bus_names_df = pd.read_excel('Data_Raw_P.xlsx', sheet_name='Sheet2')  # Adjust as necessary
    # bus_names_df = pd.read_excel('Bus_names.xlsx', sheet_name='Sheet1', header=None)
    # bus_names = bus_names_df.iloc[:, 0].tolist()  # Assuming bus names are in the first column
    
    # Step 1: Create the high_der_slope table with bus names
    create_table_query = """
    DROP TABLE IF EXISTS high_der_slope;
    CREATE TABLE high_der_slope (
        id SERIAL PRIMARY KEY,
        time_series TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Auto timestamp
        positive_slope FLOAT8,  -- Stores positive slopes
        negative_slope FLOAT8,   -- Stores negative slopes
        positive_bus_name VARCHAR,  -- Stores the corresponding bus name for positive slope
        negative_bus_name VARCHAR   -- Stores the corresponding bus name for negative slope
    );
    """
    
    # Execute the table creation query
    cursor.execute(create_table_query)
    connection.commit()
    
    # Step 2: Alter the table to ensure bus name columns exist (if not already created)
    alter_table_query = """
    ALTER TABLE high_der_slope 
    ADD COLUMN IF NOT EXISTS positive_bus_name VARCHAR,  -- Adjust the data type as necessary
    ADD COLUMN IF NOT EXISTS negative_bus_name VARCHAR;  -- Adjust the data type as necessary
    """
    cursor.execute(alter_table_query)
    connection.commit()
    
    # Step 3: Pair bus names with slopes
    bus_slopes = list(zip(bus_names, der_slope))  # Pair bus names with slopes
    
    # Step 4: Find the top 5 positive and negative slopes
    high_positive_slopes = sorted([(bus_name, slope) for bus_name, slope in bus_slopes if slope > 0], key=lambda x: x[1], reverse=True)[:5]
    high_negative_slopes = sorted([(bus_name, slope) for bus_name, slope in bus_slopes if slope < 0], key=lambda x: x[1])[:5]
    
    # Step 5: Insert the data into the database
    for pos_entry, neg_entry in zip(
        high_positive_slopes + [(None, None)] * (5 - len(high_positive_slopes)), 
        high_negative_slopes + [(None, None)] * (5 - len(high_negative_slopes))
    ):
        bus_name_pos, pos_slope = pos_entry
        bus_name_neg, neg_slope = neg_entry
        
        insert_query = """
        INSERT INTO high_der_slope (time_series, positive_slope, negative_slope, positive_bus_name, negative_bus_name)
        VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s);
        """
        
        cursor.execute(insert_query, (pos_slope, neg_slope, bus_name_pos, bus_name_neg))
    
    # Commit the changes to save the inserted data into the database
    connection.commit()
    
    
    
    
    #####Bus number
    
    bus_numbers = list(range(1, len(der_slope) + 1))  # Assuming bus numbers are sequential
    
    # Pair slopes with bus numbers
    bus_slopes = list(zip(bus_numbers, der_slope))
    
    # Filter for positive slopes and sort by slope value
    positive_slopes = sorted(
        [(bus_number, slope) for bus_number, slope in bus_slopes if slope > 0],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Get the top 1 highest positive slope
    # top_positive_slope = positive_slopes[0] if positive_slopes else None
    
    # # Now insert into the database
    # if top_positive_slope:
    #     bus_number, slope = top_positive_slope
    
    #     # Prepare to insert into the database
    #     insert_query = """
    #     INSERT INTO bus_number (time_series, bus_number, positive_slope)
    #     VALUES (CURRENT_TIMESTAMP, %s, %s);
    #     """
    #     cursor.execute(insert_query, (bus_number, slope))
    # connection.commit()
    
    top_positive_slope = positive_slopes[0] if positive_slopes else None
    
    # Explicitly define source bus number
    if top_positive_slope:
        bus_number, slope = top_positive_slope
    else:
        bus_number = None
    
    # Store source bus in database
    if bus_number is not None:
        insert_query = """
        INSERT INTO bus_number (time_series, bus_number, positive_slope)
        VALUES (CURRENT_TIMESTAMP, %s, %s);
        """
        cursor.execute(insert_query, (bus_number, slope))
        connection.commit()
    
    # MSE CALCULATION
    create_table_query = """ 
        DROP TABLE IF EXISTS mse_value;
        CREATE TABLE mse_value (
        id SERIAL PRIMARY KEY,
        Source_Bus FLOAT8,                   -- FLOAT8 for float64
        MSE_Value FLOAT8                  -- FLOAT8 for float64
    ); """
    
    cursor.execute(create_table_query)
    connection.commit()
    
    if bus_number is not None:
    
        src_bus_idx = bus_number - 1
    
        # Original Active Power (MW)
        P_source = np.real(ma_array[src_bus_idx]).flatten()
    
        # Reconstructed Prony signal (MW)
        yr_source = globals().get(f"yr_bus_{bus_number}")
    
        if yr_source is not None:
    
            # Ensure equal length
            min_len = min(len(P_source), len(yr_source))
            P_source = P_source[:min_len]
            yr_source = yr_source[:min_len]
    
            # Mean Square Error using NumPy
            mse_value = np.mean((P_source - yr_source) ** 2)
            mse_value = round(float(mse_value), 6)
            print("\n======================================")
            print(f"Mean Square Error (MSE) for Source Bus {bus_number}: {mse_value}")
            print("======================================\n")
            
            insert_query = """
            INSERT INTO mse_value (Source_Bus, MSE_Value)
            VALUES (%s, %s);
            """
            cursor.execute(insert_query, (bus_number, mse_value))
            connection.commit()
            print("MSE data inserted successsfully.")
        else:
            print("Warning: yr data not available for source bus. MSE skipped.")
    else:
        print("Warning: Source bus not identified. MSE skipped.")
    
    
    # oscillation p
    
    osci_P = ma_array[bus_number - 1].tolist()
    first_osci_P = osci_P[0]
    #Load timestamps from the Excel sheet
    # Assuming the timestamps are in the first column of the first sheet
    # timestamps_df = pd.read_excel('Data_Raw_P.xlsx', usecols='A', skiprows=0, nrows=4500)  # Adjust the file name and sheet as necessary
    
    # # Convert the timestamps to a list
    # timestamps = timestamps_df.iloc[:, 0].dropna().tolist()  # Adjust index if timestamps are in a different column
    
    create_table_query = """
    DROP TABLE IF EXISTS oscillation_data;
    CREATE TABLE oscillation_data (
        id SERIAL PRIMARY KEY,
        first_osci_P FLOAT8,
        time_series TIMESTAMP
    );
    """
    cursor.execute(create_table_query)
    connection.commit()
    
    # SQL query to insert the osci_P values and time series into the database
    insert_query = """
    INSERT INTO oscillation_data (first_osci_P, time_series)
    VALUES (%s, %s);
    """
    
    # Create a list of tuples for the insert operation
    data_to_insert = list(zip(first_osci_P, timestamps))
    
    # Bulk insert using executemany
    cursor.executemany(insert_query, data_to_insert)
    
    # Commit the transaction to save the data in the database
    connection.commit()
    
    
    
    #######mode metr
    num_rows = max(len(mode_meter_0_2), len(mode_meter_2_4), len(mode_meter_4_6), len(mode_meter_6_12))
    
    # Extend the shorter lists with None values (so all lists have the same length)
    mode_meter_0_2 += [None] * (num_rows - len(mode_meter_0_2))
    mode_meter_2_4 += [None] * (num_rows - len(mode_meter_2_4))
    mode_meter_4_6 += [None] * (num_rows - len(mode_meter_4_6))
    mode_meter_6_12 += [None] * (num_rows - len(mode_meter_6_12))
    
    # Insert query to insert values into separate columns
    insert_query = """
        INSERT INTO mode_meter_data (range_0_2, range_2_4, range_4_6, range_6_12)
        VALUES (%s, %s, %s, %s);
    """
    
    # Insert the data row by row
    for i in range(num_rows):
        cursor.execute(insert_query, (
            mode_meter_0_2[i],
            mode_meter_2_4[i],
            mode_meter_4_6[i],
            mode_meter_6_12[i]
        ))
    
    # Commit the changes to save the data in the database
    connection.commit()
    
    print("Data inserted successfully.")
    
    
    # Close the cursor and connection
    cursor.close()
    connection.close()
    
    #%% network diagram
    try:
        
        data = pd.read_excel('Rajasthan_Buses.xlsx')
        substations = data['Substation Name'].tolist()
        to_bus = data['Station ID'].tolist()
        from_bus = data['FEEDER DETAILS'].tolist()
        
        to_bus_clean = []
        for item in to_bus:
            if pd.isna(item):
                to_bus_clean.append(None)
            else:
                s = str(item)[5:][:-3]
                s = re.sub(r"_+", "", s)
                to_bus_clean.append(s)
        
        from_bus_clean = []
        for item in from_bus:
            if pd.isna(item):
                from_bus_clean.append(None)
            else:
                # s = str(item)[5:][:-3]
                s = str(item)
                s = re.sub(r"[\s\-\/\.]+", "", s)
                s = re.sub(r"_+", "", s)
                s = re.sub(r'^(L|B)\d{2,3}', '', s)
                s = re.sub(r'\d+$', '', s)
                from_bus_clean.append(s)
        
        fr_node = from_bus_clean.copy()
        to_node = to_bus_clean.copy()
        
        junk = {
            "SVC", "Transformer", "BUS",
        }
        
        fr_node_clean = [None if item in junk else item for item in fr_node]
        path = pd.DataFrame({
            'From': fr_node_clean,
            'To': to_node
        })
        df_graph = path.dropna(subset=['From', 'To'])
        df_graph = df_graph[(df_graph['From'] != '') & (df_graph['To'] != '')]
        
        
        
        defst_df = pd.DataFrame(DEFST_combine).T
        defst_df.columns = bus_names
        defst_buses = defst_df.columns.tolist()
        defst_clean = []
        for item in defst_buses:
            if pd.isna(item):
                defst_clean.append(None)
            else:
                # s = str(item)[5:][:-3]
                s = str(item)
                s = re.sub(r"[\s\-\/\.]+", "", s)
                s = re.sub(r"_+", "", s)
                s = re.sub(r'^(L|B)\d{2,3}', '', s)
                s = re.sub(r'\d+$', '', s)
                defst_clean.append(s)
        
        direction = defst_df.iloc[0] - defst_df.iloc[-1]
        
        status_values = np.sign(direction).astype(int).values
        arrow_map = pd.DataFrame({
            'Bus': defst_clean,
            'Status': status_values
        })
        status_dict = dict(zip(arrow_map['Bus'], arrow_map['Status']))
        
        #%%
        from pyvis.network import Network
        net = Network(height="600px", width="960px", notebook=True, cdn_resources='remote', bgcolor="#222222", font_color="#ffffff", directed=True)
        
        for i, row in df_graph.iterrows():
            f_bus = str(row['From'])
            t_bus = str(row['To'])
            
            net.add_node(f_bus, label=f_bus, title=f_bus, color='red', shape='square', size=30)
            net.add_node(t_bus, label=t_bus, title=t_bus, color='lightblue', shape='circle', size=30)
            
            if t_bus.upper() in status_dict:
                status = status_dict[t_bus.upper()]
                
                if status == 1:
                    net.add_edge(f_bus, t_bus, arrows='to', color="#05e705", width=2.5, title = "Flowing Into the Station")
                elif status == -1:
                    net.add_edge(t_bus, f_bus, arrows='to', color="#f54c85", width=2.5, title = "Flowing Out of the Station")
                else:
                    net.add_edge(f_bus, t_bus, arrows=None, color="#eaf820", width=2.5)
            else:
                net.add_edge(f_bus, t_bus, arrows='to', color="#FFFFFF", width=2, dashes=True)
        
        net.force_atlas_2based(gravity=-200, central_gravity=0.01, spring_length=100, spring_strength=0.1)
        print("Network diagram created.")
        net.show('Raj_net.html')
    except Exception as e:
        print(f"An error occurred in Network diagram creation!!: {e}")
except Exception as e:
    print(f"An error occurred with the core logic!!: {e}")







