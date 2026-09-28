"""Endless Shepard-tone riser for COME ON TOO STRONG.

Tempo-locked to 174 bpm, tuned to G# with a D# fifth (power-chord colour).
Every partial climbs one octave per 4 bars; 16 bars = 4 cycles, and the file
loops seamlessly, so it can run under every drop without ever arriving.

Run: python3 make_shepard_riser.py   (needs numpy, soundfile, librosa)
"""
import os
import numpy as np, soundfile as sf, librosa

SR = 44100
BPM = 174.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
TC = 4 * BAR                 # seconds per octave climb
N_CYC = 4
T = N_CYC * TC               # 16 bars
n = int(round(T * SR))
t = np.arange(n) / SR

G_SHARP_0 = 25.9565          # Hz
CENTER = np.log2(830.0)      # loudness bell centred near G#5, above the guitars
SIGMA = 1.35                 # bell width in octaves
VOICES = range(-1, 11)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shepard_riser_174bpm_Gsharp_16bars.wav')


def gain_at(f):
    """Shepard loudness bell over log-frequency, faded out before it can alias."""
    return np.exp(-0.5 * ((np.log2(f) - CENTER) / SIGMA) ** 2) * np.clip((18000.0 - f) / 2000.0, 0.0, 1.0)


def shepard_stack(base, gain, pan_sign):
    # Render one cycle at a time: at each cycle boundary voice k has reached the
    # start frequency of voice k+1, so it hands over its phase for a click-free join.
    seg_n = int(round(TC * SR))
    r = 2.0 ** (np.arange(seg_n) / SR / TC)
    up = 2 * np.pi * TC / np.log(2) * (r - 1)      # integral of 2*pi*2^(t/TC)
    phase0 = {k: 0.0 for k in VOICES}
    Ls, Rs = [], []
    for _ in range(N_CYC):
        l = np.zeros(seg_n); rr = np.zeros(seg_n)
        handover = {}
        for k in VOICES:
            f0 = base * 2.0 ** k
            s = gain * gain_at(f0 * r) * np.sin(phase0[k] + f0 * up)
            p = 0.5 + pan_sign * 0.18 * (1 if k % 2 else -1)   # alternate voices L/R
            l += s * np.sqrt(1 - p); rr += s * np.sqrt(p)
            handover[k + 1] = (phase0[k] + f0 * up[-1] + 2 * np.pi * f0 * r[-1] / SR) % (2 * np.pi)
        phase0 = {k: handover.get(k, 0.0) for k in VOICES}
        Ls.append(l); Rs.append(rr)
    L = np.concatenate(Ls); R = np.concatenate(Rs)
    return np.pad(L, (0, max(0, n - len(L))))[:n], np.pad(R, (0, max(0, n - len(R))))[:n]


L1, R1 = shepard_stack(G_SHARP_0, 1.0, +1)          # G# octaves
L2, R2 = shepard_stack(G_SHARP_0 * 1.5, 0.45, -1)   # D# fifths

# Shepard noise whoosh: noise shaped by the same gliding octave bumps. The noise is
# tiled 3x and only the middle copy kept, so the result is circular and loops cleanly.
rng = np.random.default_rng(174)
hop, nfft = 441, 4096
lf = np.log2(np.maximum(librosa.fft_frequencies(sr=SR, n_fft=nfft), 1.0))[:, None]
noise = []
for _ in range(2):
    S = librosa.stft(np.tile(rng.standard_normal(n), 3), n_fft=nfft, hop_length=hop)
    phase_in_cycle = ((np.arange(S.shape[1]) * hop / SR) % T % TC) / TC
    W = np.zeros(S.shape)
    for k in range(-1, 12):
        centre = np.log2(G_SHARP_0 * 2.0 ** k) + phase_in_cycle
        W += np.exp(-0.5 * ((lf - centre[None, :]) / 0.12) ** 2) * gain_at(2.0 ** centre)[None, :]
    noise.append(librosa.istft(S * W, hop_length=hop, length=3 * n)[n:2 * n])


def rms(x):
    return np.sqrt(np.mean(x ** 2))


tone_L, tone_R = L1 + L2, R1 + R2
noise_gain = 0.35 * rms(tone_L) / rms(noise[0])
x = np.stack([tone_L + noise_gain * noise[0], tone_R + noise_gain * noise[1]], axis=1)

# Gentle beat-locked pump (sidechain feel) so it grooves with the drop.
x *= (1.0 - 0.22 * np.exp(-9.0 * (t % BEAT) / BEAT))[:, None]

# Circular zero-phase high-pass at 180 Hz keeps it out of the reese bass and keeps the loop seamless.
X = np.fft.rfft(x, axis=0)
f = np.fft.rfftfreq(n, 1 / SR)
x = np.fft.irfft(X * ((f / 180.0) ** 4 / np.sqrt(1 + (f / 180.0) ** 8))[:, None], n=n, axis=0)

x *= 0.89 / np.abs(x).max()                         # peak about -1 dBFS
sf.write(OUT, x, SR, subtype='PCM_24')
print('wrote %s: %.3fs, %.0f bars, octave every %.3fs' % (OUT, T, T / BAR, TC))
print('loop seam step L/R: %.4f %.4f' % (abs(x[0, 0] - x[-1, 0]), abs(x[0, 1] - x[-1, 1])))
