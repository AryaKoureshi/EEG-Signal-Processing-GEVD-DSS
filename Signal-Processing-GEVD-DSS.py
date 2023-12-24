'''
Signal-Processing-GEVD-DSS

Arya Koureshi (arya.koureshi@gmail.com)

arya.koureshi@gmail.com

'''

#%% Question 1
#%% Imports
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from sklearn.decomposition import FastICA
from scipy.linalg import eigh
import warnings
warnings.filterwarnings("ignore")

#%% part a
# Load Data
data = scipy.io.loadmat('C:/Users/aryak/Downloads/Comp_HW3/Q1.mat')

X_org = data['X_org']  # The observation signal
X_1 = data['X1'] 
X_2 = data['X2'] 
X_3 = data['X3'] 
X_4 = data['X4']

T_1 = data['T1'].flatten()
T_2 = data['T2'].flatten()

# GEVD function as defined previously
def GEVD(P, C):
    eigenvalues, eigenvectors = eigh(P, C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

# Center the data by subtracting the mean
X_org_centered = X_org - np.mean(X_org, axis=1, keepdims=True)
# Estimate the cross-correlation matrix P_x for GEVD
P_x_gevd = np.dot(X_org_centered[:, :-400], X_org_centered[:, 400:].T) / (X_org.shape[1] - 400)
# Symmetrize P_x
P_x_hat_gevd = (P_x_gevd + P_x_gevd.T) / 2
# Estimate the covariance matrix C_x
C_x_gevd = np.cov(X_org)

# Perform GEVD
eigenvalues, eigenvectors = GEVD(P_x_hat_gevd, C_x_gevd)

# Extract the first eigenvector, corresponding to the highest eigenvalue
w = eigenvectors[:, 0]

# Calculate the estimated source and its effect on the observations
s_hat_1 = w.T @ X_org
x_hat_1 = np.outer(np.linalg.pinv(np.reshape(w, (1, len(w)))), s_hat_1)

# Calculate the error of the estimate
error = np.linalg.norm(X_1 - x_hat_1, 'fro') / np.linalg.norm(X_1, 'fro')

s_hat_1 = np.reshape(s_hat_1, (1, len(s_hat_1)))
# Return the estimated source, its effect on the observations and the error
s_hat_1, x_hat_1, error

# Function to plot sources with names separately
def EEGplot(signal, method, fs=False, estimated=False):
    plt.figure(figsize=(21, 14))
    num_sources = signal.shape[0] 
    
    if method==False and estimated==False:
        plt.suptitle(f'Original Channels')
        max = 0
        min = 0
        for i in range(signal.shape[0]):
            mx = np.max(signal[i])
            mn = np.min(signal[i])
            if mx >= max: max = mx
            if mn <= min: min = mn
                
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(np.arange(0, signal.shape[1]/fs, 1/fs), signal[j, :], label=f'Ch{j + 1}', c='k')
            if j+1 != signal.shape[0]:
                plt.xticks([])
            plt.ylabel(f'Ch{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]/fs])
            plt.ylim([min, max])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
        plt.xlabel("Time (sec)")
        
    if method==False and estimated==True:
        plt.suptitle(f'Estimated Channels')
        max = 0
        min = 0
        for i in range(signal.shape[0]):
            mx = np.max(signal[i])
            mn = np.min(signal[i])
            if mx >= max: max = mx
            if mn <= min: min = mn
                
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(np.arange(0, signal.shape[1]/fs, 1/fs), signal[j, :], label=f'Ch{j + 1}', c='k')
            
            plt.ylabel(f'Ch{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]/fs])
            plt.ylim([min, max])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
        plt.xlabel("Time (sec)")
        
    elif method != False:
        colors = [
                    'b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink',  # Standard colors: blue, green, red, cyan, magenta, yellow, black
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',  # Tableau Colors
                    '#1a1a1a', '#666666', '#a6a6a6', '#d9d9d9',  # Shades of gray
                    '#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#ff33ff', '#ffff99', '#9966cc', '#ff6666', '#c2c2f0', '#ffb3e6',  # Additional colors
                ]
        plt.suptitle(f'Extracted Sources ({method})')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(np.arange(0, signal.shape[1]/fs, 1/fs), signal[j, :], label=f'S{j + 1}', c=colors[j])
            plt.yticks([])
            if j+1 != signal.shape[0]:
                plt.xticks([])
            plt.ylabel(f'S{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]/fs])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
        plt.xlabel("Time (sec)")
       
    plt.tight_layout()
    plt.show();
    
    
print("\n ======================================================================================== X_org ========================================================================================\n")
EEGplot(X_org, False, 100)
print("\n ======================================================================================== X1 ========================================================================================\n")
EEGplot(X_1, False, 100)
print("\n ======================================================================================== X2 ========================================================================================\n")
EEGplot(X_2, False, 100)
print("\n ======================================================================================== X3 ========================================================================================\n")
EEGplot(X_3, False, 100)
print("\n ======================================================================================== X4 ========================================================================================\n")
EEGplot(X_4, False, 100)

print("\n ====================================================================================== X_1 Estimated ======================================================================================\n")
EEGplot(x_hat_1, False, 100, estimated=True)
print("\n ====================================================================================== S_1 Estimated ======================================================================================\n")
EEGplot(s_hat_1, False, 100, estimated=True)


# Function to calculate RRMSE
def calculate_RRMSE(actual, estimated):
    rmse = np.sqrt(np.mean((actual - estimated) ** 2))
    rms_actual = np.sqrt(np.mean(actual ** 2))
    return rmse / rms_actual

# Calculate RRMSE for x_1(t)
rrmse_x_1 = calculate_RRMSE(X_1, x_hat_1)
rrmse_x_1


# Center the data
X_org_centered = X_org - np.mean(X_org, axis=1, keepdims=True)
C_x_gevd = np.cov(X_org)

# Iterate over possible periodicities (3 to 7 seconds)
sampling_rate = 100  # 100 Hz
min_period = 3 * sampling_rate  # 3 seconds
max_period = 7 * sampling_rate  # 7 seconds

best_period = min_period
lowest_error = np.inf

for period in range(min_period, max_period + 1):
    lag_samples = period
    P_x_gevd = np.dot(X_org_centered[:, :-lag_samples], X_org_centered[:, lag_samples:].T) / (X_org.shape[1] - lag_samples)
    P_x_hat_gevd = (P_x_gevd + P_x_gevd.T) / 2

    eigenvalues, eigenvectors = GEVD(P_x_hat_gevd, C_x_gevd)
    w = eigenvectors[:, 0]

    s_hat_1 = w.T @ X_org
    x_hat_1 = np.outer(np.linalg.pinv(np.reshape(w, (1, len(w)))), s_hat_1)

    error = calculate_RRMSE(X_1, x_hat_1)

    if error < lowest_error:
        lowest_error = error
        best_period = period
        best_s_hat_1 = s_hat_1
        best_x_hat_1 = x_hat_1

best_s_hat_1 = np.reshape(best_s_hat_1, (1, len(best_s_hat_1)))
# Results
best_period_seconds = best_period / sampling_rate
print("Best Periodicity (in seconds):", best_period_seconds)
print("Lowest Error:", lowest_error)

# best_s_hat_1 and best_x_hat_1 contain the best estimates

print("\n =================================================================================== Best X_1 Estimated ====================================================================================\n")
EEGplot(best_x_hat_1, False, 100, estimated=True)
print("\n =================================================================================== Best S_1 Estimated ====================================================================================\n")
EEGplot(best_s_hat_1, False, 100, estimated=True)

# Center the data
X_org_centered = X_org - np.mean(X_org, axis=1, keepdims=True)

# Covariance matrices for 'On' and 'Off' periods
C_x_on = np.cov(X_org_centered[:, T_1 == 1])
C_x_off = np.cov(X_org_centered[:, T_1 == 0])

# Perform GEVD
eigenvalues, eigenvectors = GEVD(C_x_on, C_x_off)
w = eigenvectors[:, 0]

# Estimate s_2(t) and its effect on observations x_2(t)
s_hat_2 = w.T @ X_org
x_hat_2 = np.outer(np.linalg.pinv(np.reshape(w, (1, len(w)))), s_hat_2)

# Calculate the error of the estimate
error_s2 = calculate_RRMSE(X_2, x_hat_2)

# Results
print("Error of the estimate for s_2(t):", error_s2)
s_hat_2 = np.reshape(s_hat_2, (1, len(s_hat_2)))
# s_hat_2 and x_hat_2 contain the estimates for s_2(t) and its effect

print("\n ===================================================================================== X_2 Estimated ====================================================================================\n")
EEGplot(x_hat_2, False, 100, estimated=True)
print("\n ===================================================================================== S_2 Estimated ====================================================================================\n")
EEGplot(s_hat_2, False, 100, estimated=True)

# Center the data
X_org_centered = X_org - np.mean(X_org, axis=1, keepdims=True)

# Covariance matrix for Partial 'On' periods and overall signal
C_x_partial_on = np.cov(X_org_centered[:, T_2 == 1])
C_x_total = np.cov(X_org_centered)

# Perform GEVD
eigenvalues, eigenvectors = GEVD(C_x_partial_on, C_x_total)
w = eigenvectors[:, 0]

# Estimate s_2(t) and its effect on observations x_2(t)
s_hat_2 = w.T @ X_org
x_hat_2 = np.outer(np.linalg.pinv(np.reshape(w, (1, len(w)))), s_hat_2)

# Calculate the error of the estimate
error_s2 = calculate_RRMSE(X_2, x_hat_2)

# Results
print("Error of the estimate for s_2(t):", error_s2)
s_hat_2 = np.reshape(s_hat_2, (1, len(s_hat_2)))
# s_hat_2 and x_hat_2 contain the estimates for s_2(t) and its effect

print("\n ===================================================================================== X_2 Estimated ====================================================================================\n")
EEGplot(x_hat_2, False, 100, estimated=True)
print("\n ===================================================================================== S_2 Estimated ====================================================================================\n")
EEGplot(s_hat_2, False, 100, estimated=True)

from scipy.signal import butter, filtfilt

# Define the Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Filter parameters
fs = 100  # Sampling frequency
lowcut = 10  # Lower frequency bound (in Hz)
highcut = 15  # Upper frequency bound (in Hz)

# Apply the filter to the data
s_hat_3 = np.apply_along_axis(butter_bandpass_filter, 1, X_org, lowcut, highcut, fs)

# Estimate the effect on observations x_3(t)
x_hat_3 = s_hat_3

# Calculate the error of the estimate
error_s3 = calculate_RRMSE(X_3, x_hat_3)

# Results
print("Error of the estimate for s_3(t):", error_s3)

# s_hat_3 and x_hat_3 contain the estimates for s_3(t) and its effect



print("\n ===================================================================================== X_3 Estimated ====================================================================================\n")
EEGplot(x_hat_3, False, 100, estimated=True)

# Bandpass filter functions as defined previously
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Sampling frequency and range limits
fs = 100
low_freq_limit = 5
high_freq_limit = 25

# Error initialization
lowest_error = np.inf
best_lowcut = low_freq_limit
best_highcut = high_freq_limit

# Iterate over sub-ranges
for lowcut in range(low_freq_limit, high_freq_limit - 5, 5):
    for highcut in range(lowcut + 5, high_freq_limit, 5):
        s_hat_3 = np.apply_along_axis(butter_bandpass_filter, 1, X_org, lowcut, highcut, fs)
        x_hat_3 = s_hat_3
        error_s3 = calculate_RRMSE(X_3, x_hat_3)

        if error_s3 < lowest_error:
            lowest_error = error_s3
            best_lowcut = lowcut
            best_highcut = highcut
            best_s_hat_3 = s_hat_3
            best_x_hat_3 = x_hat_3

# Results
print("Best Frequency Range:", best_lowcut, "Hz to", best_highcut, "Hz")
print("Lowest Error of the estimate for s_3(t):", lowest_error)

# best_s_hat_3 and best_x_hat_3 contain the best estimates for s_3(t) and its effect

print("\n =================================================================================== Best X_3 Estimated ====================================================================================\n")
EEGplot(best_x_hat_3, False, 100, estimated=True)

# Whitening the data
Cx_dss = np.cov(X_org)
lambda_dss, U_dss = np.linalg.eigh(Cx_dss)
D_dss = np.dot(np.sqrt(lambda_dss**-1)*np.eye(len(np.sqrt(lambda_dss**-1))), U_dss.T)
Z_dss = np.dot(D_dss, X_org)

# Initialize random weight vector
w_dss = np.random.randn(8, 1)
Px_org1_dss = np.sqrt(np.sum(X_1**2))
Num_of_periods = X_org.shape[1] // 400

# Iterative process for DSS
e_dss = np.zeros(100)
for i in range(100):
    s1_dss = np.dot(w_dss.T, Z_dss)
    s1_periodic_dss = np.mean([s1_dss[0, j*400:(j+1)*400] for j in range(Num_of_periods)], axis=0)
    s1_hat_plus_dss = np.tile(s1_periodic_dss, (Num_of_periods, 1)).flatten()
    w_plus_dss = np.dot(Z_dss, s1_hat_plus_dss)
    w_dss = w_plus_dss / np.linalg.norm(w_plus_dss)
    w_dss = np.reshape(w_dss, (len(w_dss), 1))
    
    s1_hat_dss = np.dot(w_dss.T, Z_dss)
    X1_hat_dss = np.dot(np.linalg.pinv(D_dss), np.dot(w_dss, s1_hat_dss))

    e_dss[i] = np.sqrt(np.sum((X_1 - X1_hat_dss)**2)) / Px_org1_dss
    if i > 0 and e_dss[i] < e_dss[i-1]:
        break

# Compute RRMSE for DSS
RRMSE1_DSS = np.sqrt(np.sum((X_1 - X1_hat_dss)**2)) / Px_org1_dss

# Output RRMSE
print("RRMSE for DSS:", RRMSE1_DSS)

print("\n ====================================================================================== X_1 Estimated ======================================================================================\n")
EEGplot(X1_hat_dss, False, 100, estimated=True)
print("\n ====================================================================================== S_1 Estimated ======================================================================================\n")
EEGplot(s1_hat_dss, False, 100, estimated=True)


# Whitening the data
Cx_dss = np.cov(X_org)
lambda_dss, U_dss = np.linalg.eigh(Cx_dss)
D_dss = np.dot(np.diag(np.sqrt(lambda_dss**-1)), U_dss.T)
Z_dss = np.dot(D_dss, X_org)

# Sampling rate and range of periodicities
sampling_rate = 100  # 100 Hz
min_period_sec = 3
max_period_sec = 7
min_period_samples = min_period_sec * sampling_rate
max_period_samples = max_period_sec * sampling_rate

lowest_error = np.inf
best_period = min_period_samples
best_estimate = None

# Iterate over possible periodicities
for period in range(min_period_samples, max_period_samples + 1):
    Num_of_periods = X_org.shape[1] // period
    remaining_samples = X_org.shape[1] - Num_of_periods * period
    Px_org1_dss = np.sqrt(np.sum(X_1**2))
    
    # Initialize random weight vector
    w_dss = np.random.randn(8, 1)

    # Iterative process for DSS
    for i in range(100):
        s1_dss = np.dot(w_dss.T, Z_dss)
        s1_periodic_dss = np.mean([s1_dss[0, j*period:(j+1)*period] for j in range(Num_of_periods)], axis=0)
        s1_hat_plus_dss = np.tile(s1_periodic_dss, (Num_of_periods, 1)).flatten()

        # Adjust the length of s1_hat_plus_dss to match the length of Z_dss
        if remaining_samples > 0:
            s1_hat_plus_dss = np.append(s1_hat_plus_dss, np.zeros(remaining_samples))

        w_plus_dss = np.dot(Z_dss, s1_hat_plus_dss)
        w_dss = w_plus_dss / np.linalg.norm(w_plus_dss)
        w_dss = np.reshape(w_dss, (len(w_dss), 1))
        
        s1_hat_dss = np.dot(w_dss.T, Z_dss)
        X1_hat_dss = np.dot(np.linalg.pinv(D_dss), np.dot(w_dss, s1_hat_dss))

        error = np.sqrt(np.sum((X_1 - X1_hat_dss)**2)) / Px_org1_dss
        if error < lowest_error:
            lowest_error = error
            best_period = period
            best_estimate = X1_hat_dss

# Output results
best_period_sec = best_period / sampling_rate
print("Best Periodicity (in seconds):", best_period_sec)
print("Lowest Error for DSS:", lowest_error)


print("\n =================================================================================== Best X_1 Estimated ====================================================================================\n")
EEGplot(best_estimate, False, 100, estimated=True)
print("\n =================================================================================== Best S_1 Estimated ====================================================================================\n")
EEGplot(s1_hat_dss, False, 100, estimated=True)

# Whitening the data
Cx_dss = np.cov(X_org)
lambda_dss, U_dss = np.linalg.eigh(Cx_dss)
D_dss = np.dot(np.diag(np.sqrt(lambda_dss**-1)), U_dss.T)
Z_dss = np.dot(D_dss, X_org)

# Initialize random weight vector
w_dss = np.random.randn(8, 1)

# Separate 'On' and 'Off' periods
Z_dss_on = Z_dss[:, T_1 == 1]
Z_dss_off = Z_dss[:, T_1 == 0]

# Apply DSS for 'On' period
for i in range(100):
    s2_dss_on = np.dot(w_dss.T, Z_dss_on)
    w_plus_dss_on = np.dot(Z_dss_on, s2_dss_on.T)
    w_dss = w_plus_dss_on / np.linalg.norm(w_plus_dss_on)
    w_dss = np.reshape(w_dss, (len(w_dss), 1))

# Apply DSS for 'Off' period
for i in range(100):
    s2_dss_off = np.dot(w_dss.T, Z_dss_off)
    w_plus_dss_off = np.dot(Z_dss_off, s2_dss_off.T)
    w_dss = w_plus_dss_off / np.linalg.norm(w_plus_dss_off)
    w_dss = np.reshape(w_dss, (len(w_dss), 1))

# Combine 'On' and 'Off' period estimates
s2_hat_dss = np.dot(w_dss.T, Z_dss)
X2_hat_dss = np.dot(np.linalg.pinv(D_dss), np.dot(w_dss, s2_hat_dss))

# Compute RRMSE for DSS
RRMSE2_DSS = np.sqrt(np.sum((X_2 - X2_hat_dss)**2)) / np.sqrt(np.sum(X_2**2))

# Output RRMSE
print("RRMSE for DSS (s_2(t)):", RRMSE2_DSS)

print("\n ===================================================================================== X_2 Estimated ====================================================================================\n")
EEGplot(X2_hat_dss, False, 100, estimated=True)
print("\n ===================================================================================== S_2 Estimated ====================================================================================\n")
EEGplot(s2_hat_dss, False, 100, estimated=True)

# Whitening the data
Cx_dss = np.cov(X_org)
lambda_dss, U_dss = np.linalg.eigh(Cx_dss)
D_dss = np.dot(np.diag(np.sqrt(lambda_dss**-1)), U_dss.T)
Z_dss = np.dot(D_dss, X_org)

# Initialize random weight vector
w_dss = np.random.randn(8, 1)

# Apply DSS for 'On' periods as indicated by T_2
Z_dss_on = Z_dss[:, T_2 == 1]
for i in range(100):
    s2_dss_on = np.dot(w_dss.T, Z_dss_on)
    w_plus_dss_on = np.dot(Z_dss_on, s2_dss_on.T)
    w_dss = w_plus_dss_on / np.linalg.norm(w_plus_dss_on)
    w_dss = np.reshape(w_dss, (len(w_dss), 1))

# Estimate s_2(t) and its effect on observations x_2(t)
s2_hat_dss = np.dot(w_dss.T, Z_dss)
X2_hat_dss = np.dot(np.linalg.pinv(D_dss), np.dot(w_dss, s2_hat_dss))

# Compute RRMSE for DSS
RRMSE2_DSS = np.sqrt(np.sum((X_2 - X2_hat_dss)**2)) / np.sqrt(np.sum(X_2**2))

# Output RRMSE
print("RRMSE for DSS (s_2(t)):", RRMSE2_DSS)

print("\n ===================================================================================== X_2 Estimated ====================================================================================\n")
EEGplot(X2_hat_dss, False, 100, estimated=True)
print("\n ===================================================================================== S_2 Estimated ====================================================================================\n")
EEGplot(s2_hat_dss, False, 100, estimated=True)

# Define the Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Filter parameters
fs = 100  # Sampling frequency
lowcut = 10  # Lower frequency bound (in Hz)
highcut = 15  # Upper frequency bound (in Hz)

# Apply the filter to the data
filtered_X_org = np.apply_along_axis(butter_bandpass_filter, 1, X_org, lowcut, highcut, fs)

# Whitening the filtered data for DSS
Cx_dss_filtered = np.cov(filtered_X_org)
lambda_dss_filtered, U_dss_filtered = np.linalg.eigh(Cx_dss_filtered)
D_dss_filtered = np.dot(np.diag(np.sqrt(lambda_dss_filtered**-1)), U_dss_filtered.T)
Z_dss_filtered = np.dot(D_dss_filtered, filtered_X_org)

# Initialize random weight vector for DSS
w_dss = np.random.randn(8, 1)

# Apply DSS to the filtered data
for i in range(100):
    s3_dss = np.dot(w_dss.T, Z_dss_filtered)
    w_plus_dss = np.dot(Z_dss_filtered, s3_dss.T)
    w_dss = w_plus_dss / np.linalg.norm(w_plus_dss)
    w_dss = np.reshape(w_dss, (len(w_dss), 1))

# Estimate s_3(t) and its effect on observations x_3(t)
s3_hat_dss = np.dot(w_dss.T, Z_dss_filtered)
X3_hat_dss = np.dot(np.linalg.pinv(D_dss_filtered), np.dot(w_dss, s3_hat_dss))

# Compute RRMSE for DSS
RRMSE3_DSS = np.sqrt(np.sum((X_3 - X3_hat_dss)**2)) / np.sqrt(np.sum(X_3**2))

# Output RRMSE
print("RRMSE for DSS (s_3(t)):", RRMSE3_DSS)

print("\n ===================================================================================== X_3 Estimated ====================================================================================\n")
EEGplot(X3_hat_dss, False, 100, estimated=True)
print("\n ===================================================================================== S_3 Estimated ====================================================================================\n")
EEGplot(s3_hat_dss, False, 100, estimated=True)

# Define the Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Sampling frequency and range limits
fs = 100
low_freq_limit = 5
high_freq_limit = 25

# Iterate over sub-ranges and apply DSS
lowest_error = np.inf
best_sub_range = None
best_estimate = None

for lowcut in range(low_freq_limit, high_freq_limit - 5, 1):
    for highcut in range(lowcut + 5, high_freq_limit, 5):
        # Apply the filter for the sub-range
        filtered_X_org = np.apply_along_axis(butter_bandpass_filter, 1, X_org, lowcut, highcut, fs)

        # Whitening the filtered data for DSS
        Cx_dss_filtered = np.cov(filtered_X_org)
        lambda_dss_filtered, U_dss_filtered = np.linalg.eigh(Cx_dss_filtered)
        D_dss_filtered = np.dot(np.diag(np.sqrt(lambda_dss_filtered**-1)), U_dss_filtered.T)
        Z_dss_filtered = np.dot(D_dss_filtered, filtered_X_org)

        # Initialize random weight vector for DSS
        w_dss = np.random.randn(8, 1)

        # Apply DSS to the filtered data
        for i in range(100):
            s3_dss = np.dot(w_dss.T, Z_dss_filtered)
            w_plus_dss = np.dot(Z_dss_filtered, s3_dss.T)
            w_dss = w_plus_dss / np.linalg.norm(w_plus_dss)
            w_dss = np.reshape(w_dss, (len(w_dss), 1))

        # Estimate and error calculation
        s3_hat_dss = np.dot(w_dss.T, Z_dss_filtered)
        X3_hat_dss = np.dot(np.linalg.pinv(D_dss_filtered), np.dot(w_dss, s3_hat_dss))
        error = np.sqrt(np.sum((X_3 - X3_hat_dss)**2)) / np.sqrt(np.sum(X_3**2))

        if error < lowest_error:
            lowest_error = error
            best_sub_range = (lowcut, highcut)
            best_estimate = X3_hat_dss

# Output the best result
print("Best Frequency Range for DSS:", best_sub_range)
print("Lowest Error for DSS (s_3(t)):", lowest_error)

print("\n =================================================================================== Best X_3 Estimated ====================================================================================\n")
EEGplot(best_estimate, False, 100, estimated=True)
print("\n =================================================================================== Best S_3 Estimated ====================================================================================\n")
EEGplot(s3_hat_dss, False, 100, estimated=True)

#%% Question 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import svd, pinv, inv
from sklearn.decomposition import PCA, FastICA
import scipy.io
import random

# Load MATLAB data
mat_data = loadmat('C:/Users/aryak/Downloads/Comp_HW2/Ex2/Ex2.mat')
X_org = mat_data['X_org']
X_noise_1 = mat_data['X_noise_1']
X_noise_2 = mat_data['X_noise_2']
X_noise_3 = mat_data['X_noise_3']
X_noise_4 = mat_data['X_noise_4']
X_noise_5 = mat_data['X_noise_5']
X_noises = [X_noise_1, X_noise_2, X_noise_3, X_noise_4, X_noise_5]

# Function to plot the original and extracted sources
def plot_sources(original, extracted, method, snr):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(original.T)
    plt.title(f'Original Signal (SNR: {snr}dB)')
    plt.subplot(2, 1, 2)
    plt.plot(extracted.T)
    plt.title(f'Extracted Sources using {method} (SNR: {snr}dB)')
    plt.tight_layout()
    plt.show()
    
# Function to plot sources with names separately
def EEGplot(signal, method, snr, denoised=False):
    plt.figure(figsize=(21, 14))
    num_sources = signal.shape[0] 
    
    if method == False and snr != False and denoised==False:
        plt.suptitle(f'Noisy Original Channels (SNR: {snr}dB)')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(signal[j, :], label=f'Ch{j + 1}', c='k')
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f'Ch{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            
    elif method != False and snr != False and denoised==False:
        colors = [
                    'b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink',  # Standard colors: blue, green, red, cyan, magenta, yellow, black
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',  # Tableau Colors
                    '#1a1a1a', '#666666', '#a6a6a6', '#d9d9d9',  # Shades of gray
                    '#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#ff33ff', '#ffff99', '#9966cc', '#ff6666', '#c2c2f0', '#ffb3e6',  # Additional colors
                ]
        plt.suptitle(f'Extracted Sources ({method}, SNR: {snr}dB)')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(signal[j, :], label=f'S{j + 1}', c=colors[j])
            plt.yticks([])
            plt.xticks([])
            plt.ylabel(f'S{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            
    elif method == False and snr == False and denoised==False:
        plt.suptitle(f'Original Channels (without noise)')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(signal[j, :], label=f'Ch{j + 1}', c='k')
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f'Ch{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            
    elif denoised==True:
        plt.suptitle(f'Denoised Channels')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(signal[j, :], label=f'Ch{j + 1}', c='k')
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f'Ch{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            
    plt.tight_layout()
    plt.show();

def add_noise(original_signal, noise, snr_db):
    noise_power = np.sum(noise**2) / len(noise)
    signal_power = np.sum(original_signal**2) / len(original_signal)
    noise_factor = np.sqrt((signal_power / noise_power) * 10**(-snr_db / 10))
    noisy_signal = original_signal + noise_factor * noise
    return noisy_signal

SNRs = [-10, -20] #dB
print("\n======================================================== Original Signal (without noise) ========================================================\n")
EEGplot(X_org, False, False)

spikes_loc = np.zeros(X_org.shape)
for ch in range(len(X_org)):
    max = np.max(X_org[ch])
    th = max * 0.4
    max_temp = th
    for i in range(len(X_org[ch])-1):
        if X_org[ch, i] >= th:
            max_temp = X_org[ch, i]
            spikes_loc[ch][i] += max_temp

EEGplot(spikes_loc, False, False)

spikes_loc_developed = np.zeros(X_org.shape)
for ch in range(len(X_org)):
    max = np.max(X_org[ch])
    th = max * 0.4
    max_temp = th
    for i in range(len(X_org[ch])-1):
        if X_org[ch, i] >= th:
            max_temp = X_org[ch, i]
            spikes_loc_developed[ch][i-80:i+120] += X_org[ch, i-80:i+120]

EEGplot(spikes_loc_developed, False, False)

SNRs = [-10, -20] #dB

first_noise = X_noises[random.randint(0, 2)]
second_noise = X_noises[random.randint(3, 4)]

for snr_db in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    print("=================================================================================================================================")
    print(f"\n======================================================== SNR: {snr_db}dB ========================================================\n")
    print("=================================================================================================================================")
    print("\n======================================================== Original Signal (without noise) ========================================================\n")
    EEGplot(X_org, False, False)
    
    print("\n======================================================== First Noisy Signal ========================================================\n")
    EEGplot(X_noisy_1, False, snr_db)

    print("\n======================================================== Second Noisy Signal ========================================================\n")
    EEGplot(X_noisy_2, False, snr_db)
    
# GEVD function as defined previously
def GEVD(P, C):
    eigenvalues, eigenvectors = eigh(P, C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

spikes_loc_developed_time = np.zeros(X_org.shape)
for ch in range(len(X_org)):
    max = np.max(X_org[ch])
    th = max * 0.3
    max_temp = th
    for i in range(len(X_org[ch])-1):
        if X_org[ch, i] >= th:
            max_temp = X_org[ch, i]
            spikes_loc_developed_time[ch][i-80:i+120] += 1

for snr_db in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    print("=================================================================================================================================")
    print(f"\n======================================================== SNR: {snr_db}dB ========================================================\n")
    print("=================================================================================================================================")

    cnt = 0
    for X_noisy in [X_noisy_1, X_noisy_2]:
        if cnt == 0:
            print(f"\n======================================================== First Denoised Signal ========================================================\n")
        else:
            print(f"\n======================================================== Second Denoised Signal ========================================================\n")
            
        # Center the data by subtracting the mean
        X_noisy_centered = X_noisy - np.mean(X_noisy, axis=1, keepdims=True)
        spikes_loc_developed_time_centered = spikes_loc_developed_time - np.mean(spikes_loc_developed_time, axis=1, keepdims=True)
        
        # Estimate the covariance matrix C_x_hat
        C_x_hat_gevd = np.cov(spikes_loc_developed_time_centered)
        # Estimate the covariance matrix C_x
        C_x_gevd = np.cov(X_noisy_centered)
        
        # Perform GEVD
        eigenvalues, eigenvectors = GEVD(C_x_hat_gevd, C_x_gevd)
        
        # Extract the first eigenvector, corresponding to the highest eigenvalue
        w = eigenvectors
        
        # Calculate the estimated source and its effect on the observations
        s_hat = w.T @ X_org
        s_hat[3:, :] = 0
        x_hat = np.dot(np.linalg.pinv(w.T), s_hat)
        EEGplot(x_hat, False, False, denoised=True)
        cnt += 1
        
#DSS
for snr_db in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    print("=================================================================================================================================")
    print(f"\n======================================================== SNR: {snr_db}dB ========================================================\n")
    print("=================================================================================================================================")

    cnt = 0
    for X_noisy in [X_noisy_1, X_noisy_2]:
        if cnt == 0:
            print(f"\n======================================================== First Denoised Signal ========================================================\n")
        else:
            print(f"\n======================================================== Second Denoised Signal ========================================================\n")
            
        # Whitening the data
        Cx_dss = np.cov(X_noisy)
        lambda_dss, U_dss = np.linalg.eigh(Cx_dss)
        D_dss = np.dot(np.sqrt(lambda_dss**-1)*np.eye(len(np.sqrt(lambda_dss**-1))), U_dss.T)
        Z_dss = np.dot(D_dss, X_noisy)
        
        # Initialize random weight vector
        w_dss = np.random.randn(len(X_noisy), 1)
        Num_of_periods = np.sum(spikes_loc_developed_time[23, :])
        
        # Iterative process for DSS
        e_dss = np.zeros(100)
        for i in range(100):
            s1_dss = np.dot(w_dss.T, Z_dss)
            s1_periodic_dss = np.zeros(s1_dss.shape)
            # Averaging over periods
            for j in range(int(Num_of_periods)):
                s1_periodic_dss += (1/Num_of_periods) * s1_dss * spikes_loc_developed_time[23, :]
            w_plus_dss = np.dot(Z_dss, s1_periodic_dss.T)
            w_dss = w_plus_dss / np.linalg.norm(w_plus_dss)
            w_dss = np.reshape(w_dss, (len(w_dss), 1))
            
            s1_hat_dss = np.dot(w_dss.T, Z_dss)
            X1_hat_dss = np.dot(np.linalg.pinv(D_dss), np.dot(w_dss, s1_hat_dss))
        
            e_dss[i] = np.sqrt(np.sum((X_noisy - X1_hat_dss)**2)) / np.sqrt(np.sum((X_noisy)**2))
            if i > 0 and e_dss[i] < e_dss[i-1]:
                break
        
        # Compute RRMSE for DSS
        RRMSE1_DSS = np.sqrt(np.sum((X_noisy - X1_hat_dss)**2)) / np.sqrt(np.sum((X_noisy)**2))
        
        # Output RRMSE
        print("RRMSE for DSS:", RRMSE1_DSS)
        EEGplot(X1_hat_dss, False, False, denoised=True)
        cnt += 1

# GEVD function as defined previously
def GEVD(P, C):
    eigenvalues, eigenvectors = eigh(P, C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

spikes_loc_developed_time = np.zeros(X_org.shape)
for ch in range(len(X_org)):
    max = np.max(X_org[ch])
    th = max * 0.3
    max_temp = th
    for i in range(len(X_org[ch])-1):
        if X_org[ch, i] >= th:
            max_temp = X_org[ch, i]
            spikes_loc_developed_time[ch][i-80:i+120] += 1

print(f"\n======================================================== GEVD ========================================================\n")
for snr_db in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    print("=================================================================================================================================")
    print(f"\n======================================================== SNR: {snr_db}dB ========================================================\n")
    print("=================================================================================================================================")

    cnt = 0
    for X_noisy in [X_noisy_1, X_noisy_2]:
        if cnt == 0:
            print(f"\n======================================================== First Denoised Signal ========================================================\n")
        else:
            print(f"\n======================================================== Second Denoised Signal ========================================================\n")
            
        # Center the data by subtracting the mean
        X_noisy_centered = X_noisy - np.mean(X_noisy, axis=1, keepdims=True)
        spikes_loc_developed_time_centered = spikes_loc_developed_time - np.mean(spikes_loc_developed_time, axis=1, keepdims=True)
        
        # Estimate the covariance matrix C_x_hat
        C_x_hat_gevd = np.cov(spikes_loc_developed_time_centered)
        # Estimate the covariance matrix C_x
        C_x_gevd = np.cov(X_noisy_centered)
        
        # Perform GEVD
        eigenvalues, eigenvectors = GEVD(C_x_hat_gevd, C_x_gevd)
        
        # Extract the first eigenvector, corresponding to the highest eigenvalue
        w = eigenvectors
        
        # Calculate the estimated source and its effect on the observations
        s_hat = w.T @ X_org
        s_hat[3:, :] = 0
        x_hat = np.dot(np.linalg.pinv(w.T), s_hat)
        cnt += 1
        
        plt.figure(figsize=(20, 3))
        plt.plot(x_hat[12, :]/np.max(x_hat[12, :]), label=f'Denoised Signal (SNR={snr_db}dB)', c='r')
        plt.plot(X_org[12, :]/np.max(X_org[12, :]), label='Original Signal', lw=2, c='b')
        plt.title(f'Signal Comparison for Channel 13 (SNR={snr_db}dB)')
        plt.legend()
        plt.yticks([])
        plt.xlim([0, len(X_org[0])])
        plt.tight_layout()
        plt.show()
    
        plt.figure(figsize=(20, 3))
        plt.plot(x_hat[23, :]/np.max(x_hat[23, :]), label=f'Denoised Signal (SNR={snr_db}dB)', c='r')
        plt.plot(X_org[23, :]/np.max(X_org[23, :]), label='Original Signal', lw=2, c='b')
        plt.title(f'Signal Comparison for Channel 24 (SNR={snr_db}dB)')
        plt.legend()
        plt.yticks([])
        plt.xlim([0, len(X_org[0])])
        plt.tight_layout()
        plt.show()

print(f"\n\n\n\n======================================================== DSS ========================================================\n")
for snr_db in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    print("=================================================================================================================================")
    print(f"\n======================================================== SNR: {snr_db}dB ========================================================\n")
    print("=================================================================================================================================")

    cnt = 0
    for X_noisy in [X_noisy_1, X_noisy_2]:
        if cnt == 0:
            print(f"\n======================================================== First Denoised Signal ========================================================\n")
        else:
            print(f"\n======================================================== Second Denoised Signal ========================================================\n")
            
        # Whitening the data
        Cx_dss = np.cov(X_noisy)
        lambda_dss, U_dss = np.linalg.eigh(Cx_dss)
        D_dss = np.dot(np.sqrt(lambda_dss**-1)*np.eye(len(np.sqrt(lambda_dss**-1))), U_dss.T)
        Z_dss = np.dot(D_dss, X_noisy)
        
        # Initialize random weight vector
        w_dss = np.random.randn(len(X_noisy), 1)
        Num_of_periods = np.sum(spikes_loc_developed_time[23, :])
        
        # Iterative process for DSS
        e_dss = np.zeros(100)
        for i in range(100):
            s1_dss = np.dot(w_dss.T, Z_dss)
            s1_periodic_dss = np.zeros(s1_dss.shape)
            # Averaging over periods
            for j in range(int(Num_of_periods)):
                s1_periodic_dss += (1/Num_of_periods) * s1_dss * spikes_loc_developed_time[23, :]
            w_plus_dss = np.dot(Z_dss, s1_periodic_dss.T)
            w_dss = w_plus_dss / np.linalg.norm(w_plus_dss)
            w_dss = np.reshape(w_dss, (len(w_dss), 1))
            
            s1_hat_dss = np.dot(w_dss.T, Z_dss)
            X1_hat_dss = np.dot(np.linalg.pinv(D_dss), np.dot(w_dss, s1_hat_dss))
        
            e_dss[i] = np.sqrt(np.sum((X_noisy - X1_hat_dss)**2)) / np.sqrt(np.sum((X_noisy)**2))
            if i > 0 and e_dss[i] < e_dss[i-1]:
                break
        
        # Compute RRMSE for DSS
        RRMSE1_DSS = np.sqrt(np.sum((X_noisy - X1_hat_dss)**2)) / np.sqrt(np.sum((X_noisy)**2))
        
        # Output RRMSE
        print("RRMSE for DSS:", RRMSE1_DSS)
        cnt += 1
        
        plt.figure(figsize=(20, 3))
        plt.plot(X1_hat_dss[12, :]/np.max(X1_hat_dss[12, :]), label=f'Denoised Signal (SNR={snr_db}dB)', c='r')
        plt.plot(X_org[12, :]/np.max(X_org[12, :]), label='Original Signal', lw=2, c='b')
        plt.title(f'Signal Comparison for Channel 13 (SNR={snr_db}dB)')
        plt.legend()
        plt.yticks([])
        plt.xlim([0, len(X_org[0])])
        plt.tight_layout()
        plt.show()
    
        plt.figure(figsize=(20, 3))
        plt.plot(X1_hat_dss[23, :]/np.max(X1_hat_dss[23, :]), label=f'Denoised Signal (SNR={snr_db}dB)', c='r')
        plt.plot(X_org[23, :]/np.max(X_org[23, :]), label='Original Signal', lw=2, c='b')
        plt.title(f'Signal Comparison for Channel 24 (SNR={snr_db}dB)')
        plt.legend()
        plt.yticks([])
        plt.xlim([0, len(X_org[0])])
        plt.tight_layout()
        plt.show()
        
# GEVD function as defined previously
def GEVD(P, C):
    eigenvalues, eigenvectors = eigh(P, C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

spikes_loc_developed_time = np.zeros(X_org.shape)
for ch in range(len(X_org)):
    max = np.max(X_org[ch])
    th = max * 0.3
    max_temp = th
    for i in range(len(X_org[ch])-1):
        if X_org[ch, i] >= th:
            max_temp = X_org[ch, i]
            spikes_loc_developed_time[ch][i-80:i+120] += 1

print(f"\n======================================================== GEVD ========================================================\n")
for snr_db in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    print(f"\n======================================================== SNR: {snr_db}dB ========================================================\n")

    cnt = 0
    for X_noisy in [X_noisy_1, X_noisy_2]:
        if cnt == 0:
            print(f"\n======================================================== First Denoised Signal ========================================================\n")
        else:
            print(f"\n======================================================== Second Denoised Signal ========================================================\n")
            
        # Center the data by subtracting the mean
        X_noisy_centered = X_noisy - np.mean(X_noisy, axis=1, keepdims=True)
        spikes_loc_developed_time_centered = spikes_loc_developed_time - np.mean(spikes_loc_developed_time, axis=1, keepdims=True)
        
        # Estimate the covariance matrix C_x_hat
        C_x_hat_gevd = np.cov(spikes_loc_developed_time_centered)
        # Estimate the covariance matrix C_x
        C_x_gevd = np.cov(X_noisy_centered)
        
        # Perform GEVD
        eigenvalues, eigenvectors = GEVD(C_x_hat_gevd, C_x_gevd)
        
        # Extract the first eigenvector, corresponding to the highest eigenvalue
        w = eigenvectors
        
        # Calculate the estimated source and its effect on the observations
        s_hat = w.T @ X_org
        s_hat[3:, :] = 0
        x_hat = np.dot(np.linalg.pinv(w.T), s_hat)
        # Calculate the error of the estimate
        error = np.linalg.norm(X_org - x_hat, 'fro') / np.linalg.norm(X_org, 'fro')
        print("RRMSE for GEVD:", error)
        cnt += 1

print(f"\n\n\n\n======================================================== DSS ========================================================\n")
for snr_db in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    print(f"\n======================================================== SNR: {snr_db}dB ========================================================\n")

    cnt = 0
    for X_noisy in [X_noisy_1, X_noisy_2]:
        if cnt == 0:
            print(f"\n======================================================== First Denoised Signal ========================================================\n")
        else:
            print(f"\n======================================================== Second Denoised Signal ========================================================\n")
            
        # Whitening the data
        Cx_dss = np.cov(X_noisy)
        lambda_dss, U_dss = np.linalg.eigh(Cx_dss)
        D_dss = np.dot(np.sqrt(lambda_dss**-1)*np.eye(len(np.sqrt(lambda_dss**-1))), U_dss.T)
        Z_dss = np.dot(D_dss, X_noisy)
        
        # Initialize random weight vector
        w_dss = np.random.randn(len(X_noisy), 1)
        Num_of_periods = np.sum(spikes_loc_developed_time[23, :])
        
        # Iterative process for DSS
        e_dss = np.zeros(100)
        for i in range(100):
            s1_dss = np.dot(w_dss.T, Z_dss)
            s1_periodic_dss = np.zeros(s1_dss.shape)
            # Averaging over periods
            for j in range(int(Num_of_periods)):
                s1_periodic_dss += (1/Num_of_periods) * s1_dss * spikes_loc_developed_time[23, :]
            w_plus_dss = np.dot(Z_dss, s1_periodic_dss.T)
            w_dss = w_plus_dss / np.linalg.norm(w_plus_dss)
            w_dss = np.reshape(w_dss, (len(w_dss), 1))
            
            s1_hat_dss = np.dot(w_dss.T, Z_dss)
            X1_hat_dss = np.dot(np.linalg.pinv(D_dss), np.dot(w_dss, s1_hat_dss))
        
            e_dss[i] = np.sqrt(np.sum((X_noisy - X1_hat_dss)**2)) / np.sqrt(np.sum((X_noisy)**2))
            if i > 0 and e_dss[i] < e_dss[i-1]:
                break
        
        # Compute RRMSE for DSS
        RRMSE1_DSS = np.sqrt(np.sum((X_noisy - X1_hat_dss)**2)) / np.sqrt(np.sum((X_noisy)**2))
        
        # Output RRMSE
        print("RRMSE for DSS:", RRMSE1_DSS)
        cnt += 1

import pandas as pd

# Data from the user input
data = {
    "SNR Method": ["-10", "-20", "-10", "-20", "-10", "-20", "-10", "-20",
                   "-10", "-10", "-20", "-20", "-10", "-10", "-20", "-20"],
    "Signal Type": ["ICA - First Noisy", "ICA - First Noisy", "ICA - Second Noisy", "ICA - Second Noisy",
                    "PCA - First Noisy", "PCA - First Noisy", "PCA - Second Noisy", "PCA - Second Noisy",
                    "GEVD - First Denoised", "GEVD - Second Denoised", "GEVD - First Denoised", "GEVD - Second Denoised",
                    "DSS - First Denoised", "DSS - Second Denoised", "DSS - First Denoised", "DSS - Second Denoised"],
    "RRMSE": [0.545775, 2.822023, 0.581260, 1.184666, 1.000000, 1.000000, 0.999734, 0.999651,
              1.000361, 0.989674, 1.000107, 1.076002, 0.960480, 0.959588, 0.998827, 0.997336]
}

# Create DataFrame
df = pd.DataFrame(data)

df.head(16)  # Display the entire DataFrame
